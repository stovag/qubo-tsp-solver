from setuptools import find_packages, setup

setup(
    name='qubo_tsp_solver',
    packages=find_packages(include=['qubo_tsp_solver']),
    version='0.1.0',
    description="A toolkit for formulating QUBO models for the TSP and solving them using either Simulated Quantum Annealing or D-Wave's hardware",
    author='Evangelos Stogiannos, Theocharis Panagiotis Charalampidis, Aikaterini Maria Lazari',
    license='MIT',
    install_requires=['numpy', 'regex', 'tsplib95', 'qubovert',
                        'dwave-system','dimod', 'dwave-neal', 'dwave-inspector'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
