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


def draw_shaded_rectangles_in_goal(image, left_post, right_post, percentages):
    x1_1, y_high, x2_1, y_low = left_post
    x1_2, _, x2_2, _ = right_post

    # Create an empty image for shaded rectangles
    shaded_rectangles = np.zeros_like(image)

    # Define boundaries - remember that x1_1 ≈ x2_1 and x1_2 ≈ x2_2
    rect_width = int(abs((x1_2 + x2_2) / 2 - (x1_1 + x2_1) / 2) / 3)
    left_rect_top_left, left_rect_bottom_right = (x1_1, y_high), (x1_1 + rect_width, y_low)
    center_rect_top_left, center_rect_bottom_right = (x1_1 + rect_width, y_high), (x1_2 - rect_width, y_low)
    right_rect_top_left, right_rect_bottom_right = (x1_2 - rect_width, y_high), (x1_2, y_low)

    # Define width and height of a percentage of type 0.xx% in the given font - assuming the percentages to be correctly passed
    #(text_width, text_height), _ = cv2.getTextSize("0.33%", cv2.FONT_HERSHEY_SIMPLEX, 0.01, 1)
    #text_offset_x, text_offset_y = ((rect_width // 2) - text_width) // 2, ((y_high - y_low) // 2 - text_height) // 2
    text_offset_x, text_offset_y = rect_width // 3, (y_high - y_low) // 4

    left_text_start = (x1_1 + text_offset_x, y_low + text_offset_y)
    center_text_start = (x1_1 + rect_width + text_offset_x, y_low + text_offset_y)
    right_text_start = (x1_2 - rect_width + text_offset_x, y_low + text_offset_y)

    # Shade rectangles in the separate image
    # TODO change opacity with respect to the highest probability
    cv2.rectangle(shaded_rectangles, left_rect_top_left, left_rect_bottom_right, (255, 0, 0), -1)  # Blue
    cv2.rectangle(shaded_rectangles, center_rect_top_left, center_rect_bottom_right, (0, 255, 0), -1)  # Green
    cv2.rectangle(shaded_rectangles, right_rect_top_left, right_rect_bottom_right, (0, 0, 255), -1)  # Red

    # Add text on the shaded rectangles
    cv2.putText(image, str(percentages[0]) + "%", left_text_start, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, str(percentages[1]) + "%", center_text_start, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, str(percentages[2]) + "%", right_text_start, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add the shaded rectangles to the original image
    return cv2.addWeighted(image, 1, shaded_rectangles, 0.5, 0)


if __name__ == "__main__":
    image = cv2.imread('../data/ball_image.png')
    image = cv2.resize(image, (32 * 20, 32 * 15))

    left_post, right_post = identify_posts(image, draw=False)
    result_image = draw_shaded_rectangles_in_goal(image, left_post, right_post, [0.4, 0.2, 0.2])

    # Display the result
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
