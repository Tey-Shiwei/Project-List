import cv2
import numpy as np

class SeamCarver:
    def __init__(self, filename, out_height, out_width):
        self.out_height = out_height
        self.out_width = out_width

        # read in image
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)
        # some large number
        self.constant = 10000000

        # start
        self.seams_carving()


    def seams_carving(self):
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        if delta_col < 0:
            self.seams_removal(delta_col * -1)


    def seams_removal(self, num_pixel):
        for dummy in range(num_pixel):
            energy_map = self.calc_energy_map()
            seam_idx = self.viterbi(energy_map)

            self.delete_seam(seam_idx)


    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy



    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)

    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))

    def viterbi(self,energy_map):        
        # w(zn+1) = ln p(xn+1 | zn+1) + max zn { ln p(zn+1 | zn) + w(zn) }
        n_row, n_col = energy_map.shape
        eps = np.finfo(float).eps # to prevent log 0

        # Traverse the first row all columns
        w_arr = np.zeros(energy_map.shape)
        idx_map = np.zeros(energy_map.shape, dtype=int)
        padding = 1
        for y in range(n_row):
            for x in range(padding, n_col - padding):
                # Connected is top-left top top-right bottom-left bottom bottom-right
                if (y == 0):
                    w_arr[y][x] = np.log(energy_map[y][x]) if energy_map[y][x] != 0 else np.log(1 - eps)
                else:
                    connected_idxs = np.array(range(max(padding, x - 1), min(x + 2, n_col - padding)))
                    connected_idxs = connected_idxs[connected_idxs >= 0]
                    connected_idxs = connected_idxs[connected_idxs < n_col]
                    connected_w = w_arr[y - 1, connected_idxs[0] : connected_idxs[-1] + 1]
                    min_w = np.amin(connected_w)
                    min_idx = connected_idxs[np.where(connected_w == min_w)[0][0]]
                    w_arr[y][x] = (np.log(energy_map[y][x]) if energy_map[y][x] != 0 else np.log(1 - eps)) + w_arr[y - 1][min_idx]
                    
                    idx_map[y][x] = min_idx
                    
        min_w = np.amin(w_arr[-1][padding:-padding]) if padding > 0 else np.amin(w_arr[-1])
        min_idx = np.where(w_arr[-1] == min_w)[0][0]
        seam_idx = [min_idx]
        for y in reversed(range(1, n_row)):
            min_idx = idx_map[y][min_idx]
            seam_idx.insert(0, min_idx)
        
        # show images with seam over time
        # temp_image = self.out_image.copy().astype(np.uint8)
        # for idx, y in enumerate(range(n_row)):
        #     temp_image[y][seam_idx[idx]] = [0, 0, 255]
        # cv2.imshow("seam", temp_image)
        # cv2.waitKey(100)
        
        return seam_idx

if __name__ == '__main__':
    filename_input = 'image_input.jpg'
    filename_output = 'image_output.jpg'


    height_input, width_input = cv2.imread(filename_input).astype(np.float64).shape[: 2]

    height_output = height_input
    width_output = width_input - 30
    print('Original image size: ', height_input,width_input)

    obj = SeamCarver(filename_input, height_output, width_output)
    obj.save_result(filename_output)







