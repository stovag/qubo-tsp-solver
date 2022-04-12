from qubo_tsp_solver import qubo_tsp_solver
import numpy as np
import os

def test_qubo_form():
    adj_matrix = [
        [0, 30, 42, 12,],
        [30, 0 ,20 , 34,],
        [42, 20, 0 , 35,],
        [12, 34, 35,  0,]
    ]

    tsp_solver = qubo_tsp_solver.TSP_Solver('tsp', 'simulated')
    tsp_solver.from_matrix(adj_matrix)
    tsp_solver.formulate_qubo()
    assert type(tsp_solver.QUBO_matrix) == np.ndarray
    assert len(tsp_solver.QUBO_matrix) == 16
    assert len(tsp_solver.QUBO_matrix[0]) == 16

def test_sim_from_matrix():
    adj_matrix = [
        [0, 30, 42, 12,],
        [30, 0 ,20 , 34,],
        [42, 20, 0 , 35,],
        [12, 34, 35,  0,]
    ]

    tsp_solver = qubo_tsp_solver.TSP_Solver('tsp', 'simulated')
    tsp_solver.from_matrix(adj_matrix)
    tsp_solver.solver_qubo()

def test_qpu_from_matrix():
    adj_matrix = [
        [0, 30, 42, 12,],
        [30, 0 ,20 , 34,],
        [42, 20, 0 , 35,],
        [12, 34, 35,  0,]
    ]

    tsp_solver = qubo_tsp_solver.TSP_Solver('tsp', 'qpu', token=os.getenv('DWAVETOKEN'))
    tsp_solver.from_matrix(adj_matrix)
    tsp_solver.solver_qubo()

def test_hyb_from_matrix():
    adj_matrix = [
        [0, 30, 42, 12,],
        [30, 0 ,20 , 34,],
        [42, 20, 0 , 35,],
        [12, 34, 35,  0,]
    ]

    tsp_solver = qubo_tsp_solver.TSP_Solver('tsp', 'hybrid', token=os.getenv('DWAVETOKEN'))
    tsp_solver.from_matrix(adj_matrix)
    tsp_solver.solver_qubo()
