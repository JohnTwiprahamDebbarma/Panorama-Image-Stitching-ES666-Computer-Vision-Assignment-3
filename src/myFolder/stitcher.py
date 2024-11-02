import pdb
import glob
import cv2
import os
import numpy as np


class PanaromaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def detect_features(self, images):
        keypoints = []
        descriptors = []
        for image in images:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoint, descriptor = self.sift.detectAndCompute(gray_image, None)
            keypoints.append(keypoint)
            descriptors.append(descriptor)
        return keypoints, descriptors

    def match_features(self, descriptors):
        matches_list = []
        matcher = cv2.BFMatcher()
        for i in range(len(descriptors) - 1):
            matches = matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            matches_list.append((i, i + 1, good_matches))
        return matches_list

    def homography_matrix(self, src_pts, dst_pts):
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0, 0], src_pts[i][0, 1]
            u, v = dst_pts[i][0, 0], dst_pts[i][0, 1]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A = np.array(A)

        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H

    def homography_with_ransac(self, src_pts, dst_pts, threshold=4.0, max_iterations=10000, min_inliers=30):
        max_inliers = []
        best_H = None

        for _ in range(max_iterations):
            idx = np.random.choice(len(src_pts), 4, replace=False)
            src_sample = src_pts[idx]
            dst_sample = dst_pts[idx]

            H = self.homography_matrix(src_sample, dst_sample)

            src_pts_homogeneous = np.hstack(
                (src_pts.reshape(-1, 2), np.ones((src_pts.shape[0], 1)))
            )
            dst_pts_projected = (H @ src_pts_homogeneous.T).T
            dst_pts_projected /= dst_pts_projected[:, 2][:, np.newaxis]

            distances = np.linalg.norm(
                dst_pts_projected[:, :2] - dst_pts.reshape(-1, 2), axis=1
            )
            inliers = distances < threshold

            # Ensuring a minimum number of inliers for reliable homography matrix:
            if np.sum(inliers) > np.sum(max_inliers) and np.sum(inliers) > min_inliers:
                max_inliers = inliers
                best_H = H

        return best_H, max_inliers

    def filter_matches_with_ransac(self, keypoints, matches_list):
        filtered_matches_list = []

        for i, j, matches in matches_list:
            src_pts = np.float32(
                [keypoints[i][m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints[j][m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            H, inliers = self.homography_with_ransac(src_pts, dst_pts)
            filtered_matches = [m for k, m in enumerate(matches) if inliers[k]]
            filtered_matches_list.append((i, j, filtered_matches, H))

        return filtered_matches_list

    def draw_top_matches(self, images, keypoints, matches_list, top_n=4):
        for k in range(len(matches_list)):
            i, j, good_matches, _ = matches_list[k]
            img_matches = cv2.drawMatches(
                images[i],
                keypoints[i],
                images[j],
                keypoints[j],
                good_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imshow(f"Matches between image {i} and image {j}", img_matches)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def warp_images(self, images, matches_list, reference_img_idx, top_n=4):
        reference_img = images[reference_img_idx]
        height, width = reference_img.shape[:2]

        all_points = []
        homographies = {
            reference_img_idx: np.eye(3)
        }  # Starting with the reference img having identity homography matrix

        # Using a loop to iteratively find homographies until no new homographies can be added, thus we get the final homography matrix for each image (Sequential Homography Calculations)
        added_new_homography = True
        while added_new_homography:
            added_new_homography = False
            for i, j, _, H in matches_list:
                # Check if one of the images has a homography to the reference img
                if j not in homographies and i in homographies:
                    homographies[j] = homographies[i] @ np.linalg.inv(H)
                    added_new_homography = True
                elif i not in homographies and j in homographies:
                    homographies[i] = homographies[j] @ H
                    added_new_homography = True
                # elif i in homographies and j in homographies:
                # continue  # Both i and j already have mappings; skip this pair

        # Calculate the panorama size by transforming the image corners to find boundaries:
        for idx, H in homographies.items():
            corners = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
            ).reshape(-1, 1, 2)
            transformed_corners = self.perspective_transform(corners, H)
            all_points.append(transformed_corners)

        all_points = np.vstack(all_points).reshape(
            -1, 2
        )  # Stacking all corner points into a single array and reshaping it
        min_x, min_y = np.floor(all_points.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(all_points.max(axis=0)).astype(int)

        panorama_width = max_x - min_x
        panorama_height = max_y - min_y
        translation_offset = np.array(
            [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]]
        )  # Creating a translation matrix to shift the panorama

        panorama = np.zeros(
            (panorama_height, panorama_width, 3), dtype=np.uint8
        )  # Initializing an empty panorama canvas

        # Warping each image onto the panorama canvas:
        for idx, H in homographies.items():
            H_adjusted = translation_offset @ H
            warped_image = self.warp_image(
                images[idx], H_adjusted, (panorama_height, panorama_width)
            )

            # Mask to handle overlaps:
            mask = (warped_image > 0).astype(np.uint8)
            panorama[mask > 0] = warped_image[mask > 0]

        # Place the reference img directly onto the panorama canvas:
        panorama[-min_y : height - min_y, -min_x : width - min_x] = reference_img

        # Collect all homographies calculated for images with respect to the reference image:
        homography_matrix_list = []
        homographies = dict(
            sorted(homographies.items())
        )  # Sort the dictionary on the basis of the keys, i.e. the image indices
        for key, value in homographies.items():
            homography_matrix_list.append(value)

        return panorama, homography_matrix_list

    # Perspective transform using homography matrix
    def perspective_transform(self, points, H):
        points_homogeneous = np.hstack(
            (points.reshape(-1, 2), np.ones((points.shape[0], 1)))
        )
        transformed_points = (H @ points_homogeneous.T).T
        transformed_points /= transformed_points[:, 2][:, np.newaxis]
        return transformed_points[:, :2]

    # Warp function for each individual image using homography
    def warp_image(self, image, H, target_shape):
        height, width = target_shape
        warped_image = np.zeros((height, width, 3), dtype=image.dtype)
        H_inv = np.linalg.inv(H)

        for y in range(height):
            for x in range(width):
                target_coords = np.array([x, y, 1])
                source_coords = H_inv @ target_coords
                source_coords /= source_coords[2]

                src_x, src_y = int(source_coords[0]), int(source_coords[1])
                if (
                    0 <= src_x < image.shape[1] and 0 <= src_y < image.shape[0]
                ):  # Checking if source coordinates are within bounds
                    warped_image[y, x] = image[src_y, src_x]

        return warped_image

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + "*"))
        print("Found {} Images for stitching".format(len(all_images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here

        images1 = []

        # Read and store images
        for image_path in all_images:
            image = cv2.imread(image_path)
            if image is not None:  # Checking if the image was loaded successfully
                images1.append(image)
        # Determine the target size based on the smallest dimensions among all images
        min_height = min(image.shape[0] for image in images1)
        min_width = min(image.shape[1] for image in images1)
        if min_height < 1000 or min_width < 1000:
            target_size = (min_width, min_height)
        else:
            target_size = (
                min_width // 5,
                min_height // 5,
            )  # Reducing the size of the images for faster processing
        images = [
            cv2.resize(image, target_size) for image in images1
        ]  # Resizing all images to the target size

        keypoints, descriptors = self.detect_features(images)  # Detect features
        matches_list = self.match_features(descriptors)  # Match features
        filtered_matches_list = self.filter_matches_with_ransac(
            keypoints, matches_list
        )  # Filter matches with RANSAC
        # self.draw_top_matches(images, keypoints, filtered_matches_list, top_n=4)  # Draw the top matches
        reference_img_idx = filtered_matches_list[1][
            1
        ]  # Assuming the second image of the first pair as the reference image
        stitched_image, homography_matrix_list = self.warp_images(
            images, filtered_matches_list, reference_img_idx, top_n=4
        )  # Warp the images and create the panorama

        self.say_hi()

        return stitched_image, homography_matrix_list

    def say_hi(self):
        print("Hii From myFolder..")
