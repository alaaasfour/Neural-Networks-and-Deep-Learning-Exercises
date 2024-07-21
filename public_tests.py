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


def linear_forward_test(target):
    np.random.seed(1)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    expected_cache = (A_prev, W, b)
    expected_Z = np.array([[3.26295337, -1.23429987]])
    expected_output = (expected_Z, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error": "Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error": "Wrong output"
        },

    ]

    multiple_test(test_cases, target)


def linear_activation_forward_test(target):
    np.random.seed(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    expected_linear_cache = (A_prev, W, b)
    expected_Z = np.array([[3.43896131, -2.08938436]])
    expected_cache = (expected_linear_cache, expected_Z)
    expected_A_sigmoid = np.array([[0.96890023, 0.11013289]])
    expected_A_relu = np.array([[3.43896131, 0.]])

    expected_output_sigmoid = (expected_A_sigmoid, expected_cache)
    expected_output_relu = (expected_A_relu, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [A_prev, W, b, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error": "Datatype mismatch with sigmoid activation"
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error": "Wrong shape with sigmoid activation"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error": "Wrong output with sigmoid activation"
        },
        {
            "name": "datatype_check",
            "input": [A_prev, W, b, 'relu'],
            "expected": expected_output_relu,
            "error": "Datatype mismatch with relu activation"
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b, 'relu'],
            "expected": expected_output_relu,
            "error": "Wrong shape with relu activation"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b, 'relu'],
            "expected": expected_output_relu,
            "error": "Wrong output with relu activation"
        }
    ]

    multiple_test(test_cases, target)


def L_model_forward_test(target):
    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    expected_cache = [((np.array([[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                                  [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                                  [1.63929108, -0.4298936, 2.63128056, 0.60182225],
                                  [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                                  [0.07612761, -0.15512816, 0.63422534, 0.810655]]),
                        np.array([[0.35480861, 1.81259031, -1.3564758, -0.46363197, 0.82465384],
                                  [-1.17643148, 1.56448966, 0.71270509, -0.1810066, 0.53419953],
                                  [-0.58661296, -1.48185327, 0.85724762, 0.94309899, 0.11444143],
                                  [-0.02195668, -2.12714455, -0.83440747, -0.46550831, 0.23371059]]),
                        np.array([[1.38503523],
                                  [-0.51962709],
                                  [-0.78015214],
                                  [0.95560959]])),
                       np.array([[-5.23825714, 3.18040136, 0.4074501, -1.88612721],
                                 [-2.77358234, -0.56177316, 3.18141623, -0.99209432],
                                 [4.18500916, -1.78006909, -0.14502619, 2.72141638],
                                 [5.05850802, -1.25674082, -3.54566654, 3.82321852]])),
                      ((np.array([[0., 3.18040136, 0.4074501, 0.],
                                  [0., 0., 3.18141623, 0.],
                                  [4.18500916, 0., 0., 2.72141638],
                                  [5.05850802, 0., 0., 3.82321852]]),
                        np.array([[-0.12673638, -1.36861282, 1.21848065, -0.85750144],
                                  [-0.56147088, -1.0335199, 0.35877096, 1.07368134],
                                  [-0.37550472, 0.39636757, -0.47144628, 2.33660781]]),
                        np.array([[1.50278553],
                                  [-0.59545972],
                                  [0.52834106]])),
                       np.array([[2.2644603, 1.09971298, -2.90298027, 1.54036335],
                                 [6.33722569, -2.38116246, -4.11228806, 4.48582383],
                                 [10.37508342, -0.66591468, 1.63635185, 8.17870169]])),
                      ((np.array([[2.2644603, 1.09971298, 0., 1.54036335],
                                  [6.33722569, 0., 0., 4.48582383],
                                  [10.37508342, 0., 1.63635185, 8.17870169]]),
                        np.array([[0.9398248, 0.42628539, -0.75815703]]),
                        np.array([[-0.16236698]])),
                       np.array([[-3.19864676, 0.87117055, -1.40297864, -3.00319435]]))]
    expected_AL = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    expected_output = (expected_AL, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)
