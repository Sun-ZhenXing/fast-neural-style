import cv2
import numpy as np


class Model:
    COMPOSITION_VII = './models/eccv16/composition_vii.t7'
    LA_MUSE = './models/eccv16/la_muse.t7'
    STARRY_NIGHT = './models/eccv16/starry_night.t7'
    THE_WAVE = './models/eccv16/the_wave.t7'
    CANDY = './models/instance_norm/candy.t7'
    FEATHERS = './models/instance_norm/feathers.t7'
    LA_MUSE = './models/instance_norm/la_muse.t7'
    MOSAIC = './models/instance_norm/mosaic.t7'
    THE_SCREAM = './models/instance_norm/the_scream.t7'
    UDNIE = './models/instance_norm/udnie.t7'


def load_model(model_path: str) -> cv2.dnn.Net:
    net = cv2.dnn.readNetFromTorch(model_path)
    if net.empty():
        print('load model failed')
        exit(-1)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print('use cuda')
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    else:
        print('use cpu')
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net


def process(image: np.ndarray, net) -> np.ndarray:
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (w, h),
        (103.939, 116.779, 123.680),
        swapRB=False,
        crop=False
    )
    net.setInput(blob)
    out: np.ndarray = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0, :] += 103.939
    out[1, :] += 116.779
    out[2, :] += 123.68
    out = out.transpose(1, 2, 0)
    out = out.clip(0, 255).astype(np.uint8)
    return out


def show_frame(frame: np.ndarray, out: np.ndarray) -> int:
    merge = np.hstack((frame, out))
    cv2.imshow('demo', merge)
    key = cv2.waitKey(1) & 0xff
    if key == ord('s'):
        cv2.imwrite('./out.jpg', merge)
    return key


def main():
    print('press q to exit')
    cap = cv2.VideoCapture(0)
    net = load_model(Model.STARRY_NIGHT)
    if not cap.isOpened():
        print('open camera failed')
        exit(-1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print('frame is empty')
            break
        out = process(frame, net)
        if show_frame(frame, out) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
