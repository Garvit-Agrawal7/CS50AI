# Traffic: A Note of my Progress

### Implementing the load_data function
There are possible uncountable ways to make this code work, but here's my version
- I knew the first thing to do is to get hold of all the subdirectories inside the data set `for sub_dir in range(NUM_CATEGORIES):`, creating a for-loop (the most obvious solution)
- Then creating a path to the individual subdirectories by `path = os.path.join(data_dir, sub_dir)`
- Accessing the images inside the subdirectories using another for-loop
- Then by using the imread() function of cv2 we get a hold of the data that will be fed to the computer.
- Now to resize the image, I used the cv.resize() function and resize the image for the given size.
- Adding all the images to a list of images
- Returning the values of labels and images

### Implementing the get_model() function
- Implemented the function by understanding the code implementations from sr5
- Altered the values of the kernel, number of filters etc., to get a more accurate and efficient result
