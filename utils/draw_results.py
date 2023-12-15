import cv2
import numpy as np
from utils.utils import draw_text_with_background


def _compute_line_distance(first_line, second_line):
    # Decompose the two parameters into their coordinates
    x1_l1, y1_l1, x2_l1, y2_l1 = first_line
    x1_l2, y1_l2, x2_l2, y2_l2 = second_line
    # Check the distance among the two axis
    distance_x = (abs(x1_l2 - x1_l1) + abs(x2_l2 - x2_l1)) // 2
    distance_y = (abs(y1_l2 - y1_l1) + abs(y2_l2 - y2_l1)) // 2
    # Assume the overall distance to be equal to the average of the twos
    return (distance_x + distance_y) // 2


def _search_posts(
    lines,
    img_width,
    img_height,
    acceptability_threshold,
    horizontal_offset,
    vertical_offset,
):
    # Identify some acceptability boundaries
    left_post_x, right_post_x = int(img_width * horizontal_offset), int(
        img_width * (1 - horizontal_offset)
    )
    post_y_low, post_y_high = int(img_height * vertical_offset), int(
        img_height * (1 - vertical_offset)
    )
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


def identify_posts(
    image,
    ang_coeff_threshold=10,
    post_acceptability_threshold=150,
    horizontal_offset=0.3,
    vertical_offset=0.3,
    draw=True,
):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve Hough Line detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough Lines Transform to detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10
    )

    # Keep only long vertical lines, which are posts candidates
    long_vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < ang_coeff_threshold:
            long_vertical_lines.append(line)

    # Identify possible left and right post among all the candidates
    left_post, right_post = _search_posts(
        long_vertical_lines,
        image.shape[1],
        image.shape[0],
        post_acceptability_threshold,
        horizontal_offset,
        vertical_offset,
    )
    # Set the height to the same value for both posts
    y_start, y_end = min(left_post[3], right_post[3]), max(left_post[1], right_post[1])
    if y_end - y_start < 340:
        if y_start < 250:
            y_end = y_start + 340
        else:
            y_start = y_end - 340
    left_post[3] = y_start
    right_post[3] = y_start
    left_post[1] = y_end
    right_post[1] = y_end

    # If needed, draw posts on the original image
    if draw:
        cv2.line(
            image,
            (left_post[0], left_post[1]),
            (left_post[2], left_post[3]),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            (right_post[0], right_post[1]),
            (right_post[2], right_post[3]),
            (0, 255, 0),
            2,
        )

    return left_post, right_post


def draw_shaded_rectangles_in_goal(
    image, left_post, right_post, percentages: list[int]
):
    x1_1, y_high, x2_1, y_low = left_post
    x1_2, _, x2_2, _ = right_post

    # Define boundaries
    rect_width = int(abs((x1_2 + x2_2) / 2 - (x1_1 + x2_1) / 2) / 3)
    left_rect_top_left, left_rect_bottom_right = (x1_1, y_high), (
        x1_1 + rect_width,
        y_low,
    )
    center_rect_top_left, center_rect_bottom_right = (x1_1 + rect_width, y_high), (
        x1_2 - rect_width,
        y_low,
    )
    right_rect_top_left, right_rect_bottom_right = (x1_2 - rect_width, y_high), (
        x1_2,
        y_low,
    )

    # Define positions for text
    text_offset_x, text_offset_y = rect_width // 3, (y_high - y_low) // 4
    left_text_start = (x1_1 + text_offset_x, y_low + text_offset_y)
    center_text_start = (x1_1 + rect_width + text_offset_x, y_low + text_offset_y)
    right_text_start = (x1_2 - rect_width + text_offset_x, y_low + text_offset_y)

    # Assuming percentages is a list of three values
    max_index = percentages.index(max(percentages))  # Index of the max percentage
    min_index = percentages.index(min(percentages))  # Index of the min percentage

    print(max_index, min_index)
    print(percentages)

    # Initialize default colors (for the case where max and min are the same)
    colors = [(0, 200, 200), (0, 200, 200), (0, 200, 200)]  # All yellow initially

    if max_index != min_index:
        colors[max_index] = (0, 255, 0)  # Max percentage in green
        colors[min_index] = (0, 0, 255)  # Min percentage in red

    # Drawing and blending each rectangle individually
    rectangle_details = [
        (
            left_rect_top_left,
            left_rect_bottom_right,
            colors[0],
            percentages[0],
        ),  # Blue
        (
            center_rect_top_left,
            center_rect_bottom_right,
            colors[1],
            percentages[1],
        ),  # Green
        (
            right_rect_top_left,
            right_rect_bottom_right,
            colors[2],
            percentages[2],
        ),  # Red
    ]

    for top_left, bottom_right, color, percentage in rectangle_details:
        shaded_rectangle = np.zeros_like(image)
        cv2.rectangle(shaded_rectangle, top_left, bottom_right, color, -1)
        image = cv2.addWeighted(image, 1, shaded_rectangle, 0.5, 0)

    # Add text on the image
    text_details = [
        (left_text_start, percentages[0]),
        (center_text_start, percentages[1]),
        (right_text_start, percentages[2]),
    ]

    for text_start, percentage in text_details:
        if percentage == max(percentages):
            color = (0, 255, 0)
        elif percentage == min(percentages):
            color = (0, 0, 255)
        else:
            color = (0, 200, 200)
        draw_text_with_background(
            image,
            f"{percentage}%",
            text_start,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            thickness=2,
            color=color,
        )

    return image


def draw_shot_predictions(image, lcr_probabilities):
    left_post, right_post = identify_posts(image, draw=False)
    result_image = draw_shaded_rectangles_in_goal(
        image, left_post, right_post, lcr_probabilities
    )
    return result_image


def draw_dive_prediction(
    image,
    lr_probabilities,
    gk_box,
    length=100,
    thickness_arrow=6,
    thickness_text=2,
    text_padding=12,
):
    assert len(lr_probabilities) == 2

    arrow_color_left = (
        (0, 255, 0) if lr_probabilities[0] > lr_probabilities[1] else (0, 0, 255)
    )
    arrow_color_right = (
        (0, 255, 0) if lr_probabilities[1] > lr_probabilities[0] else (0, 0, 255)
    )

    x, y, w, h = gk_box
    y_arrows = y + h // 5 * 1
    cv2.arrowedLine(
        image, (x, y_arrows), (x - length, y_arrows), arrow_color_left, thickness_arrow
    )
    cv2.arrowedLine(
        image,
        (x + w, y_arrows),
        (x + w + length, y_arrows),
        arrow_color_right,
        thickness_arrow,
    )
    draw_text_with_background(
        image,
        f"{lr_probabilities[0]}%",
        (x - int(length / 1.25), y_arrows - text_padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        arrow_color_left,
        thickness_text,
    )
    draw_text_with_background(
        image,
        f"{lr_probabilities[1]}%",
        (x + w + int(length / 7), y_arrows - text_padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        arrow_color_right,
        thickness_text,
    )

    return image
