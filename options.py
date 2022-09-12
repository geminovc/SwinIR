import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='classical_sr', 
                    help='classical_sr, lightweight_sr, real_sr, '
                                                                 'gray_dn, color_dn, jpeg_car, color_jpeg_car')
parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
parser.add_argument('--model_path', type=str,
                    default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
args = parser.parse_args()
