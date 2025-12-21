from PIL import Image

def add_white_border(image_path, border_pixels):
    """
    Add white border to an image.
    
    Args:
        image_path (str): Path to the input image
        border_pixels (int): Number of pixels to expand as white border
    
    Returns:
        Image: Image with white border added
    """
    img = Image.open(image_path)
    
    # Create new image with white background
    new_width = img.width + 2 * border_pixels
    new_height = img.height + 2 * border_pixels
    
    new_img = Image.new('RGB', (new_width, new_height), 'gray')
    
    # Paste original image in the center
    new_img.paste(img, (border_pixels, border_pixels))
    
    return new_img

add_white_border("testcases/case01.png", 1).save("img/case01.png")
add_white_border("testcases/case02.png", 3).save("img/case02.png")
add_white_border("testcases/case03.png", 3).save("img/case03.png")
add_white_border("testcases/case04.png", 3).save("img/case04.png")
add_white_border("testcases/case05.png", 6).save("img/case05.png")