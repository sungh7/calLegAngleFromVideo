import cv2
from fastai.vision.all import *
from utils.heatmap_point import *
from utils.draw_and_cal_angle import *

learn = load_learner('./utils/legPoint.pkl')  # load trained model


def drawAngleVideo(inputs):
    vid_list = [inputs] if len(inputs[0]) == 1 else inputs

    for vid in vid_list:

        # get video basic arg
        cap = cv2.VideoCapture(vid)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # set video save path
        vid_name = vid.split('/')[-1][:-4]
        save_path = Path('./result') / vid_name
        save_path.mkdir(parents=True, exist_ok=True)

        # set video writer arg
        fps = 30
        delay = round(1000/fps)
        out_w, out_h = (640, 480)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(
            f'./result/videos/{vid_name}.avi', fourcc, fps, (out_w, out_h))

        # draw leg line and calculate angle of video
        c = 0
        while c < vid_frames:
            c += 1

            _, image = cap.read()
            im = preprocessImage(image)

            # predict keypoints
            predPoints, _, _ = learn.predict(im)
            predPoints = np.array(predPoints, dtype=np.uint)

            # naming keypoints
            knee_point = tuple(predPoints[0])
            ankle_point = tuple(predPoints[1])
            ground_point = tuple((predPoints[1][0], predPoints[0][1]))

            # pointing keypoints
            im = drawPredPoints(im, predPoints)

            # draw lines
            im = cv2.circle(im, ground_point, 2, (255, 0, 0), -1)
            im = cv2.line(im, knee_point, ankle_point,
                          color=(0, 0, 0), thickness=1)
            im = cv2.line(im, knee_point, ground_point,
                          color=(0, 0, 0), thickness=1)

            # calculate angle
            points = [knee_point, ground_point, ankle_point]
            angle = cal_angle(knee_point=knee_point, ground_point=ground_point, ankle_point=ankle_point,
                              scaling=True, origin_width=vid_w, origin_height=vid_h)

            # draw angle figure
            cv2.putText(im, "%0.3f" % float(angle),
                        org=((int(knee_point[0]+30), int(knee_point[1]-15))),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                        color=(255, 255, 255), thickness=1
                        )

            im = Image.fromarray(im)
            im = im.resize((out_w, out_h))
            im = np.array(im)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            out.write(im)

        cap.release()
        out.release()
