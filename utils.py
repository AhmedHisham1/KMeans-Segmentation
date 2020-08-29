import numpy as np

def kmeans(img, K=2, iters=10):
    # Randomly locate the K centroids.
    centroids = np.empty((K, img.shape[1]))     # centroids init shape: empty array of size Kximg.shape[1] [e.g. for Nx3 img -> centroids: Kx3]
    for i in range(K):
        centroids[i,:] = img[np.random.randint(0, high=len(img)), :]
    
    print('init centeriods: \n', centroids)

    for _ in range(iters):
        # 1) assign each pixel to its nearst centroid
        pixel_type = np.empty((img.shape[0],1), dtype=np.int32)
        for i in range(img.shape[0]):
            distance = np.empty((1,K))      # the distance of each centroid to each of the pixels
            for j in range(K):
                distance[0,j] = np.sqrt(np.sum( np.power(img[i,:] - centroids[j,:], 2) ))
            pixel_type[i,0] = np.argmin(distance).astype(np.int32)

        # 2) calculate new centroids
        for j in range(K):
            centroid_pixel_values = img[pixel_type.flatten() == j, :]
            if len(centroid_pixel_values) == 0:
                print('A centroid was assigned no pixel values, mean: ', np.mean(centroid_pixel_values, axis=0))
                print('re-initializing this centroid randomly')
                centroids[j,:] = img[np.random.randint(0, high=len(img)), :]
            else:
                centroids[j,:] = np.mean(centroid_pixel_values, axis=0)
    
    return centroids, pixel_type


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    original_img = cv2.imread('images/colors2.jpg')
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(original_img)
    # plt.show()
    print(original_img.shape)
    img = np.reshape(original_img, (-1,3))
    img = img/255
    print(img.shape)
    centroids, pixel_type = kmeans(img, K=4, iters=10)

    print(centroids)
    segmented_img = centroids[pixel_type.flatten()]
    segmented_img = segmented_img.reshape(np.array(original_img).shape)

    plt.imshow(segmented_img)
    plt.show()