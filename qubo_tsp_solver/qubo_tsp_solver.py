import re
import dimod
import tsplib95
import numpy as np
import qubovert as qv
from neal import SimulatedAnnealingSampler as SAS
import os
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import dwave.inspector
import time

class Exectime:
    def __enter__(self):
        self.time = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.time() - self.time

def normalize(A):
    max_ = max([max(i) for i in A])
    min_ = min([min(i) for i in A])
    A = [[(A[i][j] - min_) / (max_ - min_) for j in range(len(A))] for i in range(len(A))]
    return A

def print_list(l, latex=False):
    msg = ""
    for i in l:
        row = ""
        for c in i:
            if latex:
                row += str(round(c,2))+"&"
            else:
                row += str(round(c,2))+" "
        if latex:
            msg += row[:-1] + "\\\\ \n"
        else:
            msg += row + "\n"
    print(msg)

class TSP_Solver:
    def __init__(self, problem_type, solver, num_experiments=1, token=None, log_loc=None):
        self.problem_type = problem_type
        self.solver = solver
        self.num_experiments = num_experiments
        self.log_loc = log_loc
        if token == None:
            try:
                self.token = os.getenv('DWAVETOKEN')
            except:
                print("No token provided, only Simulated Annealing available")
                pass
        else:
            self.token = token


    def __str__(self):
        return str(self.problem_type) + str(self.solver)

    def from_matrix(self, adj):
        self.adj_matrix = adj
        self.n = len(adj)

    def from_problem(self, graph_file):
        self.n = int(re.search(r'\d+', graph_file).group())
        problem = tsplib95.load(graph_file)

        if self.problem_type == 'tsp':
            if problem.edge_weight_type == 'GEO':
                self.adj_matrix = self.get_adj_from_geo(problem)
            elif problem.edge_weight_type == 'EXPLICIT':
                self.adj_matrix = self.get_adj_from_explicit(problem)

        elif self.problem_type == 'hamiltonian':
            self.adj_matrix = self.get_adj_from_hamiltonian(graph_file)

    def get_adj_from_geo(self, problem):
        adj_matrix = [[0 for j in range(self.n)] for i in range(self.n)]
        k = 0
        l = 0
        node_coords = problem.node_coords
        for i in range(len(node_coords)):
            for j in range(len(node_coords)):
                adj_matrix[i][j] = ((node_coords[i+1][0] - node_coords[j+1][0]) ** 2 + (node_coords[i+1][1] - node_coords[j+1][1]) ** 2) ** (1/2)

        return adj_matrix

    def get_adj_from_explicit(self, problem):
        adj_matrix = [[0 for j in range(self.n)] for i in range(self.n)]
        k = 0
        l = 0
        for i in range(len(problem.edge_weights)):
            for j in range(len(problem.edge_weights[i])):
                adj_matrix[k][l] = problem.edge_weights[i][j]
                adj_matrix[l][k] = problem.edge_weights[i][j]
                l += 1
                if problem.edge_weights[i][j] == 0:
                    k += 1
                    l = 0
        return adj_matrix

    def get_adj_from_hamiltonian(self, graph_file):
        adj_matrix = [[0 for j in range(n)] for i in range(n)] # Create a zero-filled adjacency matrix
        with open(graph_file) as f:
            #  Create the adjacency matrix and append it to the matrices list
            for line in f:
                if len(line.strip()) != 0:
                    line_char_list = line.split(" ")
                    l = [line_char_list[i] for i in range(len(line_char_list)) if line_char_list[i] != ""]
                    l[-1] = l[-1][:-1] # The last element of l is a digit with the newline char eg. "10\n"
                    adj_matrix[int(l[0])-1][int(l[1])-1] = 1
                    adj_matrix[int(l[1])-1][int(l[0])-1] = 1

        return adj_matrix


    def formulate_qubo(self):
        if self.n > 0:
            n = self.n
            #  Initialize the auxilary matrices
            I = np.identity(n, int)

            Ineg = np.multiply(I, -1)

            O = np.zeros([n,n], int)

            J = np.triu(np.ones([n,n], int)*2 - 3 * I)

            K = np.triu(np.eye(n, k=1, dtype=int))
            K[0][n-1] = 1

            L = np.eye(n, k=1, dtype=int)
            L[n-1][0] = 1

            # Create the Mec matrix enforcing the topology of the graph
            Mec = np.asarray([[0 for j in range(n**2)] for i in range(n**2)])
            for i in range(n):
                for j in range(n):
                    if self.adj_matrix[i][j] == 0:
                        for k in range(n):
                            for l in range(n):
                                Mec[i*n+k][j*n+l] = K[k][l]
                    else:
                        for k in range(n):
                            for l in range(n):
                                Mec[i*n+k][j*n+l] = O[k][l]


            #
            Mvp = np.zeros([n**2, n**2])
            for i in range(n):
                for k in range(n):
                    for l in range(n):
                        Mvp[i*n+k][i*n+l] = J[k][l]


            #
            Mpv = np.zeros([n**2, n**2])
            for i in range(n):
                for j in range(n):
                    if i == j:
                        for k in range(n):
                            for l in range(n):
                                Mpv[i*n+k][j*n+l] = Ineg[k][l]
                    elif i<j:
                        for k in range(n):
                            for l in range(n):
                                Mpv[i*n+k][j*n+l] = I[k][l] * 2

            # Add the constraint for the weights of the graph
            if self.problem_type == 'tsp':
                # Apply min-max normalization to the adjacency matrix
                adj_matrix_norm = normalize(np.asarray(self.adj_matrix))

                # Create the matrix of the constraint for the weights of the graph
                Mw = np.zeros([n**2, n**2])
                for i in range(n):
                    for j in range(n):
                        if i<=j:
                            for k in range(n):
                                for l in range(n):
                                    Mw[i*n+k][j*n+l] = round(float(K[k][l]) * float(adj_matrix_norm[i][j]), 2)
                        else:
                            for k in range(n):
                                for l in range(n):
                                    Mw[i*n+k][j*n+l] = round(float(L[k][l]) * float(adj_matrix_norm[i][j]), 2)

            # Combine all matrices to a single QUBO matrix
            self.QUBO_matrix = Mvp + Mpv + Mec

            if self.problem_type == 'tsp':
                self.QUBO_matrix += Mw


    def pretty_solution(self, solution, value, n):
        path_matrix = []
        path = []
        flag = 0

        for i in range(0, max(solution.keys()), n):
            path_matrix.append([])
            for j in range(i, i+n):
                path_matrix[flag].append(solution[j])
            flag += 1

        for j in range(len(path_matrix)):
            for i in range(len(path_matrix)):
                if path_matrix[i][j] == 1:
                    path.append(i+1)
                    break

        return path, path_matrix


    def solve_qubo(self):
        self.formulate_qubo()
        qubo = qv.utils.matrix_to_qubo(self.QUBO_matrix)
        for i in range(self.num_experiments):
            res, sampler_name, sim_timing = self.get_dwave_response(qubo)

            # Get solutions from D-Wave responses
            dwave_solution = res.first.sample
            value = res.first.energy

            # Get pretty solutions
            pretty_path, matrix = self.pretty_solution(dwave_solution, value, self.n)

            if self.log_loc is not None:
                self.log_experiment(res, sampler_name, sim_timing)

            if self.solver == "simulated":
                print("Path found by Simulated Annealer:")
            elif self.solver == "qpu":
                print("Path found by Quatnum Annealer:")
            elif self.solver == "hybrid":
                print("Path found by Hybrid Annealer:")
            print(pretty_path)

    def get_dwave_response(self, qubo):
        sim_timing = None
        if self.solver == "simulated":
            # Use D-Wave's simulated annealer
            with Exectime() as t:
                res = SAS().sample_qubo(qubo.Q, n_reads = 1000) # Create sampler object
                sampler_name = 'SimulatedAnnealingSampler'
            sim_timing = t.time * 1000000 # convert to microseconds
            print(sim_timing)

        elif self.solver == 'qpu':
            if self.token != None:
                sampler = DWaveSampler(token=self.token)
                sampler_name = sampler.solver.name
                composite = EmbeddingComposite(sampler)
                res = composite.sample(dimod.BQM.from_qubo(qubo.Q), num_reads=1000, annealing_time=20.0)
            else:
                raise TypeError("You have not provided a D-Wave API token")

        elif self.solver == 'hybrid':
            if self.token != None:
                sampler = LeapHybridSampler(token=self.token)
                sampler_name = sampler.solver.name
                res = sampler.sample(dimod.BQM.from_qubo(qubo.Q)) # Create sampler object
            else:
                raise TypeError("You have not provided a D-Wave API token")

        else:
            raise ValueError("You have not provided an acceptable solver type ('qpu', 'hybrid' or 'simulated')")

        return res, sampler_name, sim_timing

    def log_experiment(self, res, sampler_name, sim_timing):

        if self.log_loc[-1] == "/":
            log_path = self.log_loc+"log.csv"
        else:
            log_path = self.log_loc+"/log.csv"

        if self.solver != 'simulated':
            try:
                embedding = res.info['embedding_context']['embedding']
                timing = res.info['timing']
                qpu_access_time = timing['qpu_access_time']
                run_time = qpu_access_time
                log_var = len(embedding.keys())
                qubits = sum(len(chain) for chain in embedding.values())
            except:
                qpu_access_time = res.info['qpu_access_time']
                run_time = res.info['run_time']
                log_var = 'NaN'
                qubits = 'NaN'
        else:
            embedding = 'NaN'
            qpu_access_time = 'NaN'
            run_time = sim_timing
            log_var = 'NaN'
            qubits = 'NaN'

        pretty_path, matrix = self.pretty_solution(res.first.sample, res.first.energy, self.n)
        graph_path = [pretty_path[i]-1 for i in range(len(pretty_path))]
        cost = 'NaN'
        if self.problem_type == 'tsp':
            cost = 0
            for i in range(len(graph_path)-1):
                cost += self.adj_matrix[graph_path[i]][graph_path[i+1]]
            cost += self.adj_matrix[graph_path[-1]][graph_path[0]]

        violations_code = self.get_violations_code(graph_path)

        csv_row = "\n"+str(self.n)+","+self.problem_type+","+str(log_var)+","+\
            str(qubits)+","+str(violations_code)+","+str(cost)+","+\
            str(qpu_access_time)+","+str(run_time)+","+str(sampler_name)

        if os.path.isfile(log_path):
            with open(log_path,'a') as log:
                log.write(csv_row)
        else:
            if not os.path.exists(self.log_loc):
                os.makedirs(self.log_loc)
            with open(log_path,'w') as log:
                log.write("Nodes,Type,Logical_Variables,Qubits,Violations,Cost,Qpu_Access_Time(us),Run_Time(us),Solver")
                log.write(csv_row)

    def get_violations_code(self, graph_path):
        violations_code = 0

        if len(graph_path) != len(self.adj_matrix):
            violations_code += 1

        if len(set(graph_path)) != len(graph_path):
            violations_code += 10

        for i in range(1, len(graph_path)):
            if self.adj_matrix[graph_path[i-1]][graph_path[i]] == 0 and violations_code == 0:
                violations_code += 100

        return violations_code

if __name__ == '__main__':
    adj_matrix = [
        [0, 30, 42, 12,],
        [30, 0 ,20 , 34,],
        [42, 20, 0 , 35,],
        [12, 34, 35,  0,]
    ]

    tsp_solver = TSP_Solver('tsp', 'qpu', log_loc='/home/qubo-tsp-solver-logs')
    tsp_solver.from_matrix(adj_matrix)
    tsp_solver.solve_qubo()
