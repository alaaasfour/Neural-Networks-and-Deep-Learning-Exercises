import numpy as np
from test_utils import single_test, multiple_test
         
def sigmoid_test(target):
    x = np.array([0, 2])
    output = target(x)
    assert type(output) == np.ndarray, "Wrong type. Expected np.ndarray"
    assert np.allclose(output, [0.5, 0.88079708]), f"Wrong value. {output} != [0.5, 0.88079708]"
    output = target(1)
    assert np.allclose(output, 0.7310585), f"Wrong value. {output} != 0.7310585"
    print('\033[92mAll tests passed!')
    
            
        
def initialize_with_zeros_test_1(target):
    dim = 3
    w, b = target(dim)
    assert type(b) == float, f"Wrong type for b. {type(b)} != float"
    assert b == 0., "b must be 0.0"
    assert type(w) == np.ndarray, f"Wrong type for w. {type(w)} != np.ndarray"
    assert w.shape == (dim, 1), f"Wrong shape for w. {w.shape} != {(dim, 1)}"
    assert np.allclose(w, [[0.], [0.], [0.]]), f"Wrong values for w. {w} != {[[0.], [0.], [0.]]}"
    print('\033[92mFirst test passed!')
    
def initialize_with_zeros_test_2(target):
    dim = 4
    w, b = target(dim)
    assert type(b) == float, f"Wrong type for b. {type(b)} != float"
    assert b == 0., "b must be 0.0"
    assert type(w) == np.ndarray, f"Wrong type for w. {type(w)} != np.ndarray"
    assert w.shape == (dim, 1), f"Wrong shape for w. {w.shape} != {(dim, 1)}"
    assert np.allclose(w, [[0.], [0.], [0.], [0.]]), f"Wrong values for w. {w} != {[[0.], [0.], [0.], [0.]]}"
    print('\033[92mSecond test passed!')    

def propagate_test(target):
    w, b = np.array([[1.], [2.], [-1]]), 2.5, 
    X = np.array([[1., 2., -1., 0], [3., 4., -3.2, 1], [3., 4., -3.2, -3.5]])
    Y = np.array([[1, 1, 0, 0]])

    expected_dw = np.array([[-0.03909333], [ 0.12501464], [-0.99960809]])
    expected_db = np.float64(0.288106326429569)
    expected_grads = {'dw': expected_dw,
                      'db': expected_db}
    expected_cost = np.array(2.0424567983978403)
    expected_output = (expected_grads, expected_cost)
    
    grads, cost = target( w, b, X, Y)

    assert type(grads['dw']) == np.ndarray, f"Wrong type for grads['dw']. {type(grads['dw'])} != np.ndarray"
    assert grads['dw'].shape == w.shape, f"Wrong shape for grads['dw']. {grads['dw'].shape} != {w.shape}"
    assert np.allclose(grads['dw'], expected_dw), f"Wrong values for grads['dw']. {grads['dw']} != {expected_dw}"
    assert np.allclose(grads['db'], expected_db), f"Wrong values for grads['db']. {grads['db']} != {expected_db}"
    assert np.allclose(cost, expected_cost), f"Wrong values for cost. {cost} != {expected_cost}"
    print('\033[92mAll tests passed!')

def optimize_test(target):
    w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    expected_w = np.array([[-0.70916784], [-0.42390859]])
    expected_b = np.float64(2.26891346)
    expected_params = {"w": expected_w,
                       "b": expected_b}
   
    expected_dw = np.array([[0.06188603], [-0.01407361]])
    expected_db = np.float64(-0.04709353)
    expected_grads = {"dw": expected_dw,
                      "db": expected_db}
    
    expected_cost = [5.80154532, 0.31057104]
    expected_output = (expected_params, expected_grads, expected_cost)
    
    params, grads, costs = target(w, b, X, Y, num_iterations=101, learning_rate=0.1, print_cost=False)
    
    assert type(costs) == list, "Wrong type for costs. It must be a list"
    assert len(costs) == 2, f"Wrong length for costs. {len(costs)} != 2"
    assert np.allclose(costs, expected_cost), f"Wrong values for costs. {costs} != {expected_cost}"
    
    assert type(grads['dw']) == np.ndarray, f"Wrong type for grads['dw']. {type(grads['dw'])} != np.ndarray"
    assert grads['dw'].shape == w.shape, f"Wrong shape for grads['dw']. {grads['dw'].shape} != {w.shape}"
    assert np.allclose(grads['dw'], expected_dw), f"Wrong values for grads['dw']. {grads['dw']} != {expected_dw}"
    
    assert np.allclose(grads['db'], expected_db), f"Wrong values for grads['db']. {grads['db']} != {expected_db}"
    
    assert type(params['w']) == np.ndarray, f"Wrong type for params['w']. {type(params['w'])} != np.ndarray"
    assert params['w'].shape == w.shape, f"Wrong shape for params['w']. {params['w'].shape} != {w.shape}"
    assert np.allclose(params['w'], expected_w), f"Wrong values for params['w']. {params['w']} != {expected_w}"
    
    assert np.allclose(params['b'], expected_b), f"Wrong values for params['b']. {params['b']} != {expected_b}"

    
    print('\033[92mAll tests passed!')   
        
def predict_test(target):
    w = np.array([[0.3], [0.5], [-0.2]])
    b = -0.33333
    X = np.array([[1., -0.3, 1.5],[2, 0, 1], [0, -1.5, 2]])
    
    pred = target(w, b, X)
    
    assert type(pred) == np.ndarray, f"Wrong type for pred. {type(pred)} != np.ndarray"
    assert pred.shape == (1, X.shape[1]), f"Wrong shape for pred. {pred.shape} != {(1, X.shape[1])}"
    assert np.bitwise_not(np.allclose(pred, [[1., 1., 1]])), f"Perhaps you forget to add b in the calculation of A"
    assert np.allclose(pred, [[1., 0., 1]]), f"Wrong values for pred. {pred} != {[[1., 0., 1.]]}"
    
    print('\033[92mAll tests passed!')
    
def model_test(target):
    np.random.seed(0)
    
    expected_output = {'costs': [np.array(0.69314718)], 
                   'Y_prediction_test': np.array([[1., 1., 0.]]), 
                   'Y_prediction_train': np.array([[1., 1., 0., 1., 0., 0., 1.]]), 
                   'w': np.array([[ 0.08639757],
                           [-0.08231268],
                           [-0.11798927],
                           [ 0.12866053]]), 
                   'b': -0.03983236094816321}
    
    # Use 7 samples for training
    b, Y, X = 1.5, np.array([[1, 0, 0, 1, 0, 0, 1]]), np.random.randn(4, 7),

    # Use 3 samples for testing
    x_test = np.random.randn(4, 3)
    y_test = np.array([[0, 1, 0]])

    d = target(X, Y, x_test, y_test, num_iterations=50, learning_rate=0.01)
    
    assert type(d['costs']) == list, f"Wrong type for d['costs']. {type(d['costs'])} != list"
    assert len(d['costs']) == 1, f"Wrong length for d['costs']. {len(d['costs'])} != 1"
    assert np.allclose(d['costs'], expected_output['costs']), f"Wrong values for d['costs']. {d['costs']} != {expected_output['costs']}"
    
    assert type(d['w']) == np.ndarray, f"Wrong type for d['w']. {type(d['w'])} != np.ndarray"
    assert d['w'].shape == (X.shape[0], 1), f"Wrong shape for d['w']. {d['w'].shape} != {(X.shape[0], 1)}"
    assert np.allclose(d['w'], expected_output['w']), f"Wrong values for d['w']. {d['w']} != {expected_output['w']}"
    
    assert np.allclose(d['b'], expected_output['b']), f"Wrong values for d['b']. {d['b']} != {expected_output['b']}"
    
    assert type(d['Y_prediction_test']) == np.ndarray, f"Wrong type for d['Y_prediction_test']. {type(d['Y_prediction_test'])} != np.ndarray"
    assert d['Y_prediction_test'].shape == (1, x_test.shape[1]), f"Wrong shape for d['Y_prediction_test']. {d['Y_prediction_test'].shape} != {(1, x_test.shape[1])}"
    assert np.allclose(d['Y_prediction_test'], expected_output['Y_prediction_test']), f"Wrong values for d['Y_prediction_test']. {d['Y_prediction_test']} != {expected_output['Y_prediction_test']}"
    
    assert type(d['Y_prediction_train']) == np.ndarray, f"Wrong type for d['Y_prediction_train']. {type(d['Y_prediction_train'])} != np.ndarray"
    assert d['Y_prediction_train'].shape == (1, X.shape[1]), f"Wrong shape for d['Y_prediction_train']. {d['Y_prediction_train'].shape} != {(1, X.shape[1])}"
    assert np.allclose(d['Y_prediction_train'], expected_output['Y_prediction_train']), f"Wrong values for d['Y_prediction_train']. {d['Y_prediction_train']} != {expected_output['Y_prediction_train']}"
    
    print('\033[92mAll tests passed!')


def initialize_parameters_test_1(target):
    n_x, n_h, n_y = 3, 2, 1
    expected_W1 = np.array([[0.01624345, -0.00611756, -0.00528172], [-0.01072969, 0.00865408, -0.02301539]])
    expected_b1 = np.array([[0.], [0.]])
    expected_W2 = np.array([[0.01744812, -0.00761207]])
    expected_b2 = np.array([[0.]])
    expected_output = {"W1": expected_W1,
                       "b1": expected_b1,
                       "W2": expected_W2,
                       "b2": expected_b2}
    test_cases = [
        {
            "name": "datatype_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    multiple_test(test_cases, target)


def initialize_parameters_test_2(target):
    n_x, n_h, n_y = 4, 3, 2

    expected_W1 = np.array([[0.01624345, -0.00611756, -0.00528172, -0.01072969],
                            [0.00865408, -0.02301539, 0.01744812, -0.00761207],
                            [0.00319039, -0.0024937, 0.01462108, -0.02060141]])
    expected_b1 = np.array([[0.], [0.], [0.]])
    expected_W2 = np.array([[-0.00322417, -0.00384054, 0.01133769],
                            [-0.01099891, -0.00172428, -0.00877858]])
    expected_b2 = np.array([[0.], [0.]])
    expected_output = {"W1": expected_W1,
                       "b1": expected_b1,
                       "W2": expected_W2,
                       "b2": expected_b2}
    test_cases = [
        {
            "name": "datatype_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    multiple_test(test_cases, target)

def initialize_parameters_deep_test_1(target):
    layer_dims = [5, 4, 3]
    expected_W1 = np.array([[0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
                            [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
                            [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
                            [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]])
    expected_b1 = np.array([[0.],
                            [0.],
                            [0.],
                            [0.]])
    expected_W2 = np.array([[-0.01185047, -0.0020565, 0.01486148, 0.00236716],
                            [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
                            [-0.00768836, -0.00230031, 0.00745056, 0.01976111]])
    expected_b2 = np.array([[0.],
                            [0.],
                            [0.]])
    expected_output = {"W1": expected_W1,
                       "b1": expected_b1,
                       "W2": expected_W2,
                       "b2": expected_b2}
    test_cases = [
        {
            "name": "datatype_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)

def initialize_parameters_deep_test_2(target):
    layer_dims = [4, 3, 2]
    expected_W1 = np.array([[0.01788628, 0.0043651, 0.00096497, -0.01863493],
                            [-0.00277388, -0.00354759, -0.00082741, -0.00627001],
                            [-0.00043818, -0.00477218, -0.01313865, 0.00884622]])
    expected_b1 = np.array([[0.],
                            [0.],
                            [0.]])
    expected_W2 = np.array([[0.00881318, 0.01709573, 0.00050034],
                            [-0.00404677, -0.0054536, -0.01546477]])
    expected_b2 = np.array([[0.],
                            [0.]])
    expected_output = {"W1": expected_W1,
                       "b1": expected_b1,
                       "W2": expected_W2,
                       "b2": expected_b2}
    test_cases = [
        {
            "name": "datatype_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)
