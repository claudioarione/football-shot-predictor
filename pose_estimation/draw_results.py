import cv2
import numpy as np


def _compute_line_distance(first_line, second_line):
    # Decompose the two parameters into their coordinates
    x1_l1, y1_l1, x2_l1, y2_l1 = first_line
    x1_l2, y1_l2, x2_l2, y2_l2 = second_line
    # Check the distance among the two axis
    distance_x = (abs(x1_l2 - x1_l1) + abs(x2_l2 - x2_l1)) // 2
    distance_y = (abs(y1_l2 - y1_l1) + abs(y2_l2 - y2_l1)) // 2
    # Assume the overall distance to be equal to the average of the twos
    return (distance_x + distance_y) // 2


def _search_posts(lines, img_width, img_height, acceptability_threshold, horizontal_offset, vertical_offset):
    # Identify some acceptability boundaries
    left_post_x, right_post_x = int(img_width * horizontal_offset), int(img_width * (1 - horizontal_offset))
    post_y_low, post_y_high = int(img_height * vertical_offset), int(img_height * (1 - vertical_offset))
    # First estimate the position of the two posts
    estimated_left_post = [left_post_x, post_y_high, left_post_x, post_y_low]
    estimated_right_post = [right_post_x, post_y_high, right_post_x, post_y_low]
    # Set the thresholds
    acc_thresh_left, acc_thresh_right = acceptability_threshold, acceptability_threshold
    # Search left post in the left third of the image and right post similarly
    for line in lines:
        if _compute_line_distance(line[0], estimated_left_post) < acc_thresh_left:
            estimated_left_post = line[0]
            acc_thresh_left //= 2
        elif _compute_line_distance(line[0], estimated_right_post) < acc_thresh_right:
            estimated_right_post = line[0]
            acc_thresh_right //= 2

    # Return the found values
    return estimated_left_post, estimated_right_post


def identify_posts(image, ang_coeff_threshold=10, post_acceptability_threshold=60,
                   horizontal_offset=0.2, vertical_offset=0.3, draw=True):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve Hough Line detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough Lines Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    # Keep only long vertical lines, which are posts candidates
    long_vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < ang_coeff_threshold:
            long_vertical_lines.append(line)

    # Identify possible left and right post among all the candidates
    left_post, right_post = _search_posts(long_vertical_lines, image.shape[1], image.shape[0],
                                          post_acceptability_threshold, horizontal_offset, vertical_offset)
    # Set the height to the same value for both posts
    y_start, y_end = min(left_post[3], right_post[3]), max(left_post[1], right_post[1])
    left_post[3] = y_start
    right_post[3] = y_start
    left_post[1] = y_end
    right_post[1] = y_end

    # If needed, draw posts on the original image
    if draw:
        cv2.line(image, (left_post[0], left_post[1]), (left_post[2], left_post[3]), (0, 255, 0), 2)
        cv2.line(image, (right_post[0], right_post[1]), (right_post[2], right_post[3]), (0, 255, 0), 2)

    return left_post, right_post


if __name__ == "__main__":
    image = cv2.imread('../data/ball_image.png')
    image = cv2.resize(image, (32 * 20, 32 * 15))

    left_post, right_post = identify_posts(image)

    # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
