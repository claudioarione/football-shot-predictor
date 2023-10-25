import cv2
import cvzone
import cvzone.ColorModule as cm

def load_video():
    # Load the video
    cap = cv2.VideoCapture('data/Penalty_Neymar.mp4')
    hsv_vals = {
            "hmin": 0, "smin": 24, "vmin": 200, "hmax": 50, "smax": 61, "vmax": 255
        }
    color_finder = cm.ColorFinder(False)
    while True:
        # Read the image
        success, frame = cap.read()
        # if not success:
        #     break

        # frame = cv2.imread("data/ball_image.png")
        # Resize the image putting the max size to 800 - TODO remove?
        frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)

        # Find the ball inside the screen
        # frame_color, mask = color_finder.update(frame, {
        #     "hmin": 31, "smin": 0, "vmin": 200, "hmax": 43, "smax": 61, "vmax": 255
        # })
        frame_color, mask = color_finder.update(frame, hsv_vals)  # FIXME: find better values because for the video are not good
        # frame_color, mask = color_finder.update(frame, "red")  # to find the right hsv_vals
        # Find ball location
        # frame_contours, contours = cvzone.findContours(frame, mask, minArea=250)
        # Show the ball frame
        # ball_frame = cv2.resize(ball_frame, (0, 0), None, 0.7, 0.7)
        # cv2.imshow('Ball frame', ball_frame)

        cv2.imshow('Frame', frame)
        cv2.imshow('frameColor', frame_color)

        if cv2.waitKey(100) & 0xFF == ord('q'):  # FIXME: quit
            break

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    load_video()
