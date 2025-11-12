import torch
import qtensor as ql
import numpy as np
from quant_model import QLinearLayer, QModel, emit_c_array, emit_c_float, tensor2str
from include.SmallCNN import load_data, transform, evaluate, SmallCNN
import pickle
import os
import shutil

IN_FEATURES = 50
CALIB_STEPS = 1000
TEST_STEPS = 1000

def test_one_linear_layer():
    print("Testing standalone Quantized Linear Layer...")
    layer = torch.nn.Linear(IN_FEATURES, 10)
    qlayer = QLinearLayer(layer)

    for i in range(CALIB_STEPS):
        x = torch.randn(1, IN_FEATURES)
        qlayer.calibrate(x)

    qlayer.quantize()

    y1 = layer(x)
    y2 = qlayer.forward(x)

    print(f"y1:\n{y1}")
    print(f"y2:\n{y2}")


def test_quant_for_linear_model(model):
    model.eval()
    qmodel = QModel(model)

    # Calibrate the model's quantization parameters
    for i in range(CALIB_STEPS):
        x = torch.randn(1, IN_FEATURES)
        qmodel.calibrate(x)

    # Quantize the model
    qmodel.quantize()

    # Evaluate
    correct = 0
    total = 0
    for i in range(TEST_STEPS):
        x_test = torch.randn(1, IN_FEATURES)
        y1 = model(x_test).detach()
        y2 = qmodel.forward(x_test)

        if (np.argmax(y1) == np.argmax(y2)):
            correct += 1
        total += 1

    accuracy = (correct / float(total)) * 100
    print(f"Relative Accuracy: {accuracy:.2f}%")


def test_multi_linear_layers():
    print("Testing multiple Quantized Linear Layers...")

    model = torch.nn.Sequential(
        torch.nn.Linear(IN_FEATURES, 25, True),
        torch.nn.Linear(25, 25, True),
        torch.nn.Linear(25, 10, True)
    )
    test_quant_for_linear_model(model)


def test_conv2d_layers():
    print("Testing Conv2D Quantized Layers...")

    in_channels = 3
    out_channels = 10
    h = 16
    w = 16

    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 5, (3, 3), padding=1),
        torch.nn.Conv2d(5, 5, (3, 3), padding=1),
        torch.nn.Conv2d(5, out_channels, (3, 3), padding=1)
    )
    model.eval()

    qmodel = QModel(model)

    # Calibrate the model's quantization parameters
    for i in range(CALIB_STEPS):
        x = torch.randn(in_channels, h, w)
        qmodel.calibrate(x)

    # Quantize the model
    qmodel.quantize()

    total = 0
    correct = 0
    for i in range(TEST_STEPS):
        x_test = torch.randn(in_channels, h, w)
        y1 = model(x_test).detach()
        y2 = qmodel.forward(x_test)

        range_y1 = y1.max() - y1.min()
        atol = max(0.05 * range_y1.item(), 1e-4)  # tolerance = 5% of output range, at least 1e-4

        # Check if outputs match +- tolerance
        if torch.allclose(y1, y2, atol=atol) == True:
            correct += 1
        total += 1
    
    accuracy = (correct / float(total)) * 100
    print(f"Relative Accuracy: {accuracy:.2f}%")


def test_export_linear():
    EXPORT_DIR='./export/'

    load = 0 # set to 1 when loading existing model ( for comparison with HLS ), set to 0 when initializing new weights
    x = torch.tensor([ -0.679435670375824, 1.1476980447769165, -0.8773130774497986, -1.6388133764266968, -0.9310069680213928, 1.0613003969192505, -1.966984748840332, -0.5416407585144043, -0.8141457438468933, -0.37850645184516907, -1.904554843902588, 0.8769300580024719, -0.2594682276248932, 0.08589325845241547, 1.08950936794281, -1.5660877227783203, 0.646989107131958, -0.21247076988220215, -0.8272263407707214, -0.8937994241714478, 0.6280809044837952, -1.0904560089111328, 1.009001612663269, -0.32792848348617554, -1.4629429578781128, 0.06571771949529648, -1.0842418670654297, -1.6674408912658691, 2.0854763984680176, 1.2472699880599976, 1.014737606048584, -0.7465372681617737, 1.2965564727783203, 1.0371558666229248, -1.6650141477584839, 0.4480772018432617, 0.07346311956644058, -0.40100622177124023, 0.8673726320266724, 0.8060222864151001, 0.12725307047367096, 1.0295659303665161, -0.11763858795166016, 0.7851923108100891, 1.1535874605178833, -0.3013613224029541, -0.7126262784004211, -0.18613103032112122, 1.9151064157485962, 0.784883439540863])
    print(f"x: {x}")
    # x = torch.randn(1, IN_FEATURES)

    if (load):
        with open('qmodel.pkl', 'rb') as file:
            qmodel = pickle.load(file)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(IN_FEATURES, 25, True)
        )
        model.eval()
        qmodel = QModel(model)

        # Calibrate the model's quantization parameters
        for i in range(CALIB_STEPS):
            cx = torch.randn(1, IN_FEATURES)
            qmodel.calibrate(cx)

        # Quantize the model
        qmodel.quantize()

        with open('qmodel.pkl', 'wb') as file:
            pickle.dump(qmodel, file)

    if (load == 0):
        # clean previous export
        if os.path.exists(EXPORT_DIR):
            shutil.rmtree(EXPORT_DIR)

        os.mkdir(EXPORT_DIR)

        # export first layer
        qmodel.children[0].export(EXPORT_DIR)


    y = qmodel.forward(x)
    print(f"y: {y}")

    if (load == 0):
        # export test vectors
        filepath = os.path.join(EXPORT_DIR, 'test_vectors.h')
        with open(filepath, 'w') as f:
            f.write("#include \"common.hpp\"\n\n")
            emit_c_array(f, "test_x", x, "float")
            emit_c_array(f, "test_y", y, "float")

        print("Export finished successfully!")


def test_export_conv2d():
    EXPORT_DIR='./export/'

    load = 0 # set to 1 when loading existing model ( for comparison with HLS ), set to 0 when initializing new weights

    IN_CHANNELS = 1
    OUT_CHANNELS = 3
    IMG_H = 4
    IMG_W = 4

    x = torch.rand(IN_CHANNELS, IMG_H, IMG_W)
    print(f"x: {x.shape}")

    if (load):
        with open('qmodel.pkl', 'rb') as file:
            qmodel = pickle.load(file)
    else:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, (3, 3), padding=1)
        )
        model.eval()
        qmodel = QModel(model)

        # Calibrate the model's quantization parameters
        for i in range(CALIB_STEPS):
            cx = torch.rand(IN_CHANNELS, IMG_H, IMG_W)
            qmodel.calibrate(cx)

        # Quantize the model
        qmodel.quantize()

        with open('qmodel.pkl', 'wb') as file:
            pickle.dump(qmodel, file)

    if (load == 0):
        # clean previous export
        if os.path.exists(EXPORT_DIR):
            shutil.rmtree(EXPORT_DIR)

        os.mkdir(EXPORT_DIR)

        # export first layer
        qmodel.children[0].export(EXPORT_DIR)


    y = qmodel.forward(x)
    y = y.squeeze()
    print(f"y: {y}")

    if (load == 0):
        # export test vectors
        filepath = os.path.join(EXPORT_DIR, f'{qmodel.children[0].name}_test_vectors.h')
        with open(filepath, 'w') as f:
            f.write("#include \"common.hpp\"\n\n")
            emit_c_array(f, "test_x", x, "float")
            
            print(f"y.shape; {y.shape}")
            emit_c_array(f, "test_y", y, "float")

        print("Export finished successfully!")


def test_relu_function():
    N = 100

    model = torch.nn.Sequential(torch.nn.ReLU())
    model.eval()
    qmodel = QModel(model)

    for i in range(CALIB_STEPS):
        cx = torch.rand(N)
        qmodel.calibrate(cx)

    qmodel.quantize()

    total = 0
    correct = 0
    for i in range(TEST_STEPS):
        x_test = torch.rand(N)
        y1 = model(x_test).detach()
        y2 = qmodel.forward(x_test)

        if i == 0:
            print(f"y1: {y1[:5]}\ny2: {y2[:5]}")

        range_y1 = y1.max() - y1.min()
        atol = max(0.05 * range_y1.item(), 1e-4)  # tolerance = 5% of output range, at least 1e-4

        # Check if outputs match +- tolerance
        if torch.allclose(y1, y2, atol=atol) == True:
            correct += 1
        total += 1

    accuracy = (correct / float(total)) * 100
    print(f"Relative Accuracy: {accuracy:.2f}%")

    if (correct == total):
        print("ReLU Test: PASSED")
    else:
        print("ReLU Test: FAILED")


def test_full_model():
    export_dir = "export_full"
    device = torch.device('cpu')
    torch.manual_seed(42)

    C = 3
    H = 64
    W = 64

    model = SmallCNN()

    # Load model checkpoint
    model_path = '../ai_experiments/openset_classification_experiment_2025_06_22.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    _, _, test_loader = load_data(transform=transform)

    # qmodel = QModel(model.network)
    qmodel = QLinearLayer(0, "linear", model.network[-1]) # only the last layer
    # qmodel = QModel(torch.nn.Sequential(model.network[-1])) # only the last layer
    base_model = model.network[:-1]

    for i in range(CALIB_STEPS):
        cx = torch.randn(1, C, H, W) 
        cx = base_model(cx)
        qmodel.calibrate(cx)

    qmodel.quantize()

    # if os.path.exists(export_dir):
    #     shutil.rmtree(export_dir)

    # os.mkdir(export_dir)

    best_threshold = 3.5834
    evaluate(qmodel, base_model, device, test_loader, best_threshold)

    # qmodel.export(export_dir)

    # print(f"Linear layer of model exported to {export_dir} successfully!")


test_full_model()


# train_loader, val_loader, test_loader = load_data(transform=transform)
# print("Length of train set: ", len(train_loader.dataset))
# print("Length of validation set: ", len(val_loader.dataset))
# print("Length of test set: ", len(test_loader.dataset))
