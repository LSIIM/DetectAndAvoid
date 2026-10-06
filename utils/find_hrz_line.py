import cv2 as cv
import numpy as np

def find_hrz_line(image, upper_limit=0.80):
    """
    Detects the y-coordinate of a horizontal line in the given image based on the mean pixel intensity profile.

    Args:
        image (numpy.ndarray): Input image.
        upper_limit (float): Upper limit for the y-coordinate of the horizontal line.
    Returns:
        y_limit (int): The y-coordinate of the detected horizontal line.
    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    y_min = int(h * upper_limit)

    perfil = np.mean(gray, axis=1)

    diff = np.abs(np.diff(perfil))

    y_limite = y_min + np.argmax(diff[y_min:h])

    return y_limite

if __name__ == "__main__":
    # Example usage
    image = cv.VideoCapture("/video/path")
    
    while True:
        ret, frame = image.read()
        if not ret:
            break

        frame = cv.resize(frame, (640, 480))
        height, width, _ = frame.shape


        y_limit = find_hrz_line(frame,65)
        frame[y_limit:height, :] = 0


        cv.imshow("Detected Horizontal Line", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    cv.destroyAllWindows()