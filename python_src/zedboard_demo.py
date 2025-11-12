#!/usr/bin/env python3
import os, mmap, struct, numpy as np
import errno
import json
import time
# import torch

EXPORT_DIR = "export_eval/"

TILE_H = 300
TILE_W = 10

INPUT_SIZE = 4096
OUTPUT_SIZE = 10

# ─────────────────────────  FIFO device paths  ──────────────────────────
FIFO_INPUT_DEV  = "/dev/axis_fifo_0x00000000a0000000"
FIFO_OUTPUT_DEV = "/dev/axis_fifo_0x00000000a0001000"
FIFO_WEIGHT_DEV = "/dev/axis_fifo_0x00000000a0002000"

# ────────────────────────  single-word write helper  ────────────────────
_FMT_TABLE = {np.dtype(np.float32): ("<f", 4),   # little-endian float32
              np.dtype(np.int32):    ("<i", 4)}    # signed int32


def write_val(fmt: str, value, size, fd, idx: int = None):
    """Write one value and print any I/O error immediately."""
    try:
        n = os.write(fd, struct.pack(fmt, value))
        if n != size:
            print(f"[WARN] short write {n}/{size}  idx={idx}")
    except OSError as e:
        print(f"[ERR ] idx={idx} errno={e.errno} ({e.strerror})")
        raise

# ────────────────────────  burst helpers (keep fd open)  ────────────────
def fifo_write_array(arr: np.ndarray, path: str, label: str, debug=False):
    """Stream the array value-by-value, reporting errors."""
    dt = arr.dtype
    if dt not in _FMT_TABLE:
        raise TypeError(f"Unsupported dtype {dt}; supported: {list(_FMT_TABLE)}")
        
    fmt, size = _FMT_TABLE[dt]
    
    fd = os.open(path, os.O_WRONLY)
    try:
        for i, val in enumerate(arr):
            write_val(fmt, val, size, fd, i)
    finally:
        os.close(fd)
        
    if (debug):
        print(f"[ OK ] {label}: wrote {len(arr)} {dt} words")

def fifo_read_array(n_elems: int, path: str, label: str,
                    dtype=np.int32) -> np.ndarray:
    """Read *n_elems* words from *path* and return a NumPy array of *dtype*."""
    itemsize = np.dtype(dtype).itemsize
    wanted   = n_elems * itemsize

    fd = os.open(path, os.O_RDONLY | os.O_SYNC)
    try:
        raw = bytearray()
        while len(raw) < wanted:                 # keep pulling until full
            try:
                chunk = os.read(fd, wanted - len(raw))
            except BlockingIOError as e:
                if e.errno == errno.EAGAIN:      # FIFO empty – wait/retry
                    continue
                raise
            if not chunk:                        # unexpected EOF
                break
            raw.extend(chunk)
    finally:
        os.close(fd)

    if len(raw) != wanted:
        print(f"[WARN] short read {len(raw)}/{wanted} bytes from {label}")

    return np.frombuffer(raw, dtype=dtype)

# ───────────────────────────  AXI register poke  ────────────────────────
def start_ai_model(in_scale, debug=False):
    AI_BASE_ADDR              = 0x00_A001_A000
    AI_CTRL_REGISTER          = 0x00
    AI_IN_SCALE_REGISTER      = 0x10
    AP_START          = 1 << 0

    f = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    virt = mmap.mmap(f, 0x10000,
                     flags=mmap.MAP_SHARED,
                     prot=mmap.PROT_READ | mmap.PROT_WRITE,
                     offset=AI_BASE_ADDR)
    
    # Set input scale
    virt.seek(AI_IN_SCALE_REGISTER)
    virt.write(struct.pack("=f", in_scale))
    
    # Start the ip
    virt.seek(AI_CTRL_REGISTER)
    virt.write(struct.pack("=I", AP_START))
    virt.close()
    
    os.close(f)
    
    if (debug):
        print("[ OK ] AI core kicked (AP_START)")


def fifo_soft_reset(base_addr: int, reset_tx: bool = True, reset_rx: bool = True, debug=False):
    f = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    try:
        virt = mmap.mmap(f, 0x10000, mmap.MAP_SHARED,
                         prot=mmap.PROT_READ | mmap.PROT_WRITE,
                         offset=base_addr & ~0xFFFF)
        if reset_tx:
            virt.seek((base_addr & 0xFFFF) + 0x18)
            virt.write(struct.pack("<I", 1))
        if reset_rx:
            virt.seek((base_addr & 0xFFFF) + 0x1C)
            virt.write(struct.pack("<I", 1))
    finally:
        virt.close()
        os.close(f)
        
    if (debug):
        print("[ OK ] FIFO soft-reset done")
    
def quantize(arr_f32: np.ndarray, scale: float, dtype) -> np.ndarray:
    if arr_f32.dtype != np.float32:
        raise TypeError("arr_f32 dtype must be float32")
    
    q = np.round(arr_f32 / scale)
    q = np.clip(q, -128, 127)
    return q.astype(dtype)

def dequantize(arr, scale):
    return (arr * scale).astype(np.float32)

# def dequantize():

def hw_matmul(inputs, qweights, in_scale, debug=False):   
    # -- type & dtype checks
    if not isinstance(inputs, np.ndarray) or not isinstance(qweights, np.ndarray):
        raise TypeError("Both inputs and weights must be numpy arrays")
    if inputs.dtype != np.float32:
        raise TypeError("Input dtype must be float32")
    if qweights.dtype != np.int32:
        raise TypeError("Weights dtype must be int32")

    # -- shape checks
    if inputs.shape != (TILE_H,):
        raise ValueError(f"inputs.shape must be ({TILE_H},), got {inputs.shape}")
    if qweights.shape != (TILE_H, TILE_W):
        raise ValueError(f"weights.shape must be ({INPUT_SIZE}, {OUTPUT_SIZE}), got {qweights.shape}")
        
    fifo_soft_reset(0x00_A000_0000, debug)
    fifo_soft_reset(0x00_A000_1000, debug)
    fifo_soft_reset(0x00_A000_2000, debug)
    
    start_ai_model(in_scale, debug)
    
    fifo_write_array(inputs.ravel(),   FIFO_INPUT_DEV,  "input", debug)
    fifo_write_array(qweights.ravel(),  FIFO_WEIGHT_DEV, "weight matrix", debug)
    
    outputs = fifo_read_array(OUTPUT_SIZE, FIFO_OUTPUT_DEV, "output", dtype='<i4')
    
    return outputs
    
def golden_ref(inputs, weights_q, in_scale):
    if weights_q.dtype != np.int32:
        raise TypeError("Weights dtype must be int32")
    
    inputs_q = quantize(inputs, in_scale, np.int8)
    outputs_pt = np.matmul(inputs_q.astype(np.int32), weights_q.astype(np.int32)).astype(np.int32)
    
    return outputs_pt
        
def hw_linear_layer(inputs, qweights, qbias, in_scale, bias_scale):
    pad_len = (TILE_H - INPUT_SIZE % TILE_H) % TILE_H
    if pad_len:
        inputs_pad  = np.pad(inputs,  (0, pad_len),           'constant')
        weights_pad = np.pad(qweights, ((0, pad_len), (0, 0)), 'constant')
    else:
        inputs_pad, weights_pad = inputs, qweights

    input_tiles_pad  = [inputs_pad[i:i+TILE_H]             for i in range(0, inputs_pad.size, TILE_H)]
    weight_tiles_pad = [weights_pad[i:i+TILE_H, :]         for i in range(0, inputs_pad.size, TILE_H)]

    outputs = np.zeros(OUTPUT_SIZE, dtype=np.float32)

    for x_tile, w_tile in zip(input_tiles_pad, weight_tiles_pad):   # or *_drop / *_pad
        outputs += hw_matmul(x_tile, w_tile, in_scale)       # (300,) · (300,10) → (10,)
    
    outputs += qbias
    
    return dequantize(outputs, bias_scale)


# def ip_test():
#     inputs  = np.load("input_300.npy")        # shape (300,)
#     weights  = np.load("weights_300x10.npy")   # shape (300, 10)
#     expected_output = np.load("output_10.npy")        # shape (10,)
    
#     in_scale = 0.016063753763834634;
#     weights_scale = 0.0035328075755387545; 
    
#     weights_q = quantize(weights, weights_scale, np.int8).astype(np.int32)

#     outputs = hw_matmul(inputs, weights_q, in_scale)
#     outputs_pt = golden_ref(inputs, weights_q, in_scale)
    
#     print("\nResults preview:")
#     print("inputs [0:8 ]:", inputs[0:8])
#     print("weights [0][0:8]:", weights[0][0:8])
#     print("outputs:", outputs)
#     print("expected_output:", expected_output)
#     print("outputs (pytorch):", outputs_pt)
    
#     if np.array_equal(outputs, expected_output) and np.array_equal(expected_output, outputs_pt):
#         print("IP Test PASSED")
#     else:
#         print("IP Test FAILED, outputs differ")
    
    
def big_vector_test():    
    in_scale = 0.016063753763834634
    weights_scale = 0.0035328075755387545
    
    inputs  = np.random.randn(INPUT_SIZE).astype(np.float32)              # shape (4096,)
    weights = np.random.randn(INPUT_SIZE, OUTPUT_SIZE).astype(np.float32) # shape (4096, 10)
    weights_q = quantize(weights, weights_scale, np.int8).astype(np.int32)

    print(inputs.shape, inputs.dtype)   # (4096,) float32
    print(weights_q.shape, weights_q.dtype) # (4096, 10) int32
    
    pad_len = (TILE_H - INPUT_SIZE % TILE_H) % TILE_H  # 104 padded rows
    if pad_len:
        inputs_pad  = np.pad(inputs,  (0, pad_len),           'constant')
        weights_pad = np.pad(weights_q, ((0, pad_len), (0, 0)), 'constant')
    else:
        inputs_pad, weights_pad = inputs, weights_q

    input_tiles_pad  = [inputs_pad[i:i+TILE_H]             for i in range(0, inputs_pad.size, TILE_H)]
    weight_tiles_pad = [weights_pad[i:i+TILE_H, :]         for i in range(0, inputs_pad.size, TILE_H)]
    
    outputs = np.zeros(OUTPUT_SIZE, dtype=np.int32)

    for x_tile, w_tile in zip(input_tiles_pad, weight_tiles_pad):   # or *_drop / *_pad
        outputs += hw_matmul(x_tile, w_tile, in_scale)       # (300,) · (300,10) → (10,)
    
    
    expected_output = golden_ref(inputs, weights_q, in_scale)

    print("outputs:", outputs)
    print("expected_output:", expected_output)

    if np.array_equal(outputs, expected_output):
        print("Big Vector Test PASSED")
    else:
        print("Big Vector Test FAILED, outputs differ")
            
            
def test_2():
#     np.random.seed(0)
    
    in_scale = 0.016063753763834634
    weights_scale = 0.0035328075755387545
    
    
    inputs  = np.random.randn(TILE_H).astype(np.float32)
    weights = np.random.randn(TILE_H, TILE_W).astype(np.float32)
    
    weights_q = quantize(weights, weights_scale, np.int8).astype(np.int32)
    
    outputs = hw_matmul(inputs, weights_q, in_scale, True)
    expected_output = golden_ref(inputs, weights_q, in_scale)

    print(inputs.shape, inputs.dtype)
    print(weights_q.shape, weights_q.dtype)

    print("outputs:", outputs)
    print("expected_output:", expected_output)

    if np.array_equal(outputs, expected_output):
        print("Test 2 PASSED")
    else:
        print("Test 2 FAILED, outputs differ")
        
def compute_per_class_f1_precision_recall_support(y_pred, y_true, the_class):
    tp = np.sum((y_true == the_class) & (y_pred == the_class))
    fp = np.sum((y_true != the_class) & (y_pred == the_class))
    fn = np.sum((y_true == the_class) & (y_pred != the_class))
    support = np.sum((y_true == the_class))
    
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp / float(tp + fp)
    
    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp / float(tp + fn)
        
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1_score, support

def compute_weighted_f1(y_pred, y_true):
    classes = np.unique(y_true)
    weighted_f1 = 0
    total_sup = 0

    for c in classes:
        _, _, f1, sup = compute_per_class_f1_precision_recall_support(y_pred, y_true, c)
        weighted_f1 += sup * f1
        total_sup += sup
    
    if (total_sup == 0):
        return 0
    
    return weighted_f1 / float(total_sup)
    

def compute_metrics(labels_class, labels_osr, logits, mls_thresh):
    logits = np.array(logits)
    N = logits.shape[0] # limit evaluation to first N results computed
    labels_class = labels_class[:N]
    labels_osr = labels_osr[:N]
    
    mls = np.max(logits, axis=1)
    y_pred_osr = (mls > mls_thresh).astype(int)

    y_pred_class = np.argmax(logits, axis=1)
    mask = (labels_osr == 1) # compare for known classes only
    
    print(logits.shape)
    print(y_pred_class.shape)
    print(mask.shape)
    
    labels_class_filtered = labels_class[mask]
    y_pred_filtered = y_pred_class[mask]

    f1_score_osr = compute_weighted_f1(y_pred_osr, labels_osr)
    print(f"f1_score (osr): {f1_score_osr:.4f}")

    f1_score_class = compute_weighted_f1(y_pred_filtered, labels_class_filtered)
    print(f"f1_score (classification) = {f1_score_class:.4f}")
    
    
def test_eval_on_set():
    inputs = np.load(EXPORT_DIR + "transformed_images.npy").astype(np.float32)
    qweights = np.load(EXPORT_DIR + "qweights.npy").astype(np.int32)
    qbias = np.load(EXPORT_DIR + "qbias.npy").astype(np.int32)
    labels_class = np.load(EXPORT_DIR + "labels_class.npy")
    labels_osr = np.load(EXPORT_DIR + "labels_osr.npy")
    
    with open(EXPORT_DIR + 'scales.json') as json_file:
        scales = json.load(json_file)

    in_scale = scales["input_scale"]
    weights_scale = scales["weights_scale"]
    bias_scale = scales["biases_scale"]
    mls_thresh = 3.005
    
    n = len(labels_osr)
    
    logits = []
    
    print("Evaluating on test set...")
    
    start_time = time.time()
    
    for i in range(n):
        output = hw_linear_layer(inputs[i], qweights, qbias, in_scale, bias_scale)
        logits.append(output)

        if (i%5 == 0):
            print(f"i: {i}")
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Evaluation finished, elapsed time: {elapsed_time:.4f}")
    
    compute_metrics(labels_class, labels_osr, logits, mls_thresh)

    
def demo_osr():
    inputs = np.load(EXPORT_DIR + "transformed_images.npy").astype(np.float32)
    qweights = np.load(EXPORT_DIR + "qweights.npy").astype(np.int32)
    qbias = np.load(EXPORT_DIR + "qbias.npy").astype(np.int32)
    labels_class = np.load(EXPORT_DIR + "labels_class.npy")
    labels_osr = np.load(EXPORT_DIR + "labels_osr.npy")
    
    with open(EXPORT_DIR + 'scales.json') as json_file:
        scales = json.load(json_file)

    in_scale = scales["input_scale"]
    weights_scale = scales["weights_scale"]
    bias_scale = scales["biases_scale"]
    mls_thresh = 3.005
    
    n = 5
    
    logits = []
    
    print("Demonstration Started")
    print("MLS Threshold:", mls_thresh)
    
    np.random.seed(0)
    idxs = np.random.randint(0, len(labels_osr)-1, n)
    
    print("-----------------")
    for i in idxs:
        print(f"Image #{i}")
        output = hw_linear_layer(inputs[i], qweights, qbias, in_scale, bias_scale)
        print(output)
        
        mls = np.max(output)
        print("MLS:", mls)
        
        osr_pred = mls > mls_thresh
        print("Is Known? (predicted):", osr_pred)
        print("Is Known? (actual):", bool(labels_osr[i]))
        
        if osr_pred == True: 
            y_pred = np.argmax(output)
            print("Class (predicted):", y_pred)
        print("Class (actual):", labels_class[i])
        print("-----------------")

    print("Demonstration finished")

    
# ──────────────────────────────  main demo  ─────────────────────────────
def main():   
#     ip_test()
#     test_2()
#     big_vector_test()
    test_eval_on_set()
    # demo_osr()

if __name__ == "__main__":
    main()
