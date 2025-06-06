import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler 

class TimeSeriesDataGenerator:
    def __init__(self, w, t, a, b, mod, normalize_abnormal=False):
        self.w = w
        self.t = t
        self.a = a
        self.b = b
        self.mod = mod
        self.normalize_abnormal = normalize_abnormal 
        self.data = None
        self.labels = None
        self.binary = False
        self.abnormal_std = 1
        self.abnormal_mean = 0

    def generate_binary_class_data(self, abtype):
        s = np.arange(1, self.w + 1)
        data1 = np.random.randn(self.a, self.w) * self.abnormal_std + self.abnormal_mean

        data2 = np.zeros((self.b, self.w))
        for i in range(self.b):
            if abtype == 1:  # Uptrend
                data2[i] = self.abnormal_mean + np.random.randn(self.w) * self.abnormal_std + self.t * s
            elif abtype == 2:  # Downtrend
                data2[i] = self.abnormal_mean + np.random.randn(self.w) * self.abnormal_std - self.t * s
            elif abtype == 3:  # Upshift
                shift_value = self.t  # Define the shift magnitude
                shift_point = self.w // 2  # Define the point where the shift starts (middle of the window)
                data2[i, :shift_point] = np.random.randn(shift_point) * self.abnormal_std + self.abnormal_mean  # No shift in the first half
                data2[i, shift_point:] = np.random.randn(self.w - shift_point) * self.abnormal_std + self.abnormal_mean + shift_value  # Shift starts in the second half
            
            elif abtype == 4:  # Downshift
                shift_value = self.t  # Define the shift magnitude
                shift_point = self.w // 2  # Define the point where the shift starts (middle of the window)
                data2[i, :shift_point] = np.random.randn(shift_point) * self.abnormal_std + self.abnormal_mean  # No shift in the first half
                data2[i, shift_point:] = np.random.randn(self.w - shift_point) * self.abnormal_std + self.abnormal_mean - shift_value  # Downshift starts in the second half
            
            elif abtype == 5:  # Systematic
                data2[i] = self.abnormal_mean + np.random.randn(self.w) * self.abnormal_std + self.t * (-1) ** s  # Alternating pattern
            
            elif abtype == 6:  # Cyclic
                period = 8  # Define a default period (can be made variable)
                data2[i] = self.abnormal_mean + np.random.randn(self.w) * self.abnormal_std + self.t * np.sin(2 * np.pi * s / period)
            
            elif abtype == 7:  # Stratification
                data2[i] = self.abnormal_mean + np.random.randn(self.w) * (self.t)   # Reduced variability (stratified data)
                # self.t is used as the standard deviation here to control the variability of the stratified data

        # Normalize abnormal data if needed
        if self.normalize_abnormal:
            for i in range(self.b):
                data2[i] = self.normalize_sample(data2[i])
            
        data = np.vstack((data1, data2))
        labels = np.hstack((np.ones(self.a) * -1, np.ones(self.b) * 1))
        data = np.hstack((data, labels.reshape(-1, 1)))
        
        self.data = data
        self.labels = labels
        self.binary = True


    def normalize_sample(self, sample):
        norm = np.linalg.norm(sample)
        if norm == 0:
            return sample
        return sample / norm

    def get_data(self):
        if self.data is None:
            raise ValueError("Data not generated. Please call generate_data() first.")
        return self.data[:, :-1]

    def get_labels(self):
        if self.labels is None:
            raise ValueError("Data not generated. Please call generate_data() first.")
        return self.labels

    def visualize_data(self, abtypes_to_visualize=None):
        if self.data is None:
            raise ValueError("Data not generated. Please call generate_data() first.")
        
        plt.figure(figsize=(10, 6))

        abtype_names = {
            1: 'Uptrend',
            2: 'Downtrend',
            3: 'Upshift',
            4: 'Downshift',
            5: 'Systematic',
            6: 'Cyclic',
            7: 'Stratification'
        }

        if self.binary:
            label_names = {0: 'Normal', 1: 'Abnormal'}
        else:
            if abtypes_to_visualize is None:
                abtypes_to_visualize = list(abtype_names.keys())
            label_names = {0: 'Normal'}
            for idx in abtypes_to_visualize:
                label_names[idx] = f'Abnormal {abtype_names[idx]}'

        colors = plt.cm.get_cmap('tab10', len(label_names))

        for i in range(len(self.labels)):
            label = int(self.data[i, -1])
            if label in label_names:
                plt.plot(self.data[i, :-1], color=colors(label), alpha=0.3, label=label_names[label] if i == 0 else "")

        handles = []
        for label, name in label_names.items():
            handles.append(plt.Line2D([0], [0], color=colors(label), lw=2, label=name))
        plt.legend(handles=handles)

        plt.title('Time Series Data Visualization')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()


def save_libsvm_format(data, labels, filename):
    with open(filename, 'w') as f:
        for i in range(data.shape[0]):
            label = int(labels[i])
            features = ' '.join(f"{j+1}:{data[i,j]:.6f}" for j in range(data.shape[1]))
            f.write(f"{label} {features}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic time series data.')
    parser.add_argument('-t', '--type', choices=['bc', 'mc'], required=True, help='Type of classification: bc for binary class, mc for multiclass.')
    parser.add_argument('-d', '--data_path', required=True, help='Path to save the generated data.')
    parser.add_argument('-w', '--window_length', type=int, default=48, help='Window length for time series data.')
    parser.add_argument('--t', type=float, default=0.5, help='Parameter of abnormal pattern.')
    parser.add_argument('-a', type=int, default=20, help='Size of Normal class.')
    parser.add_argument('-b', type=int, default=10, help='Size of abnormal class.')
    parser.add_argument('-m', '--mod', type=int, choices=[1, 2], default=1, help='Mode: 1 for SVM, 2 for WSVM.')
    parser.add_argument('--abtype', type=int, choices=[1, 2, 3, 4, 5, 6, 7], help='Type of abnormal pattern for binary classification.')
    parser.add_argument('--normalize_abnormal', action='store_true', help='Normalize abnormal data.')
    
    args = parser.parse_args()

    generator = TimeSeriesDataGenerator(w=args.window_length, t=args.t, a=args.a, b=args.b, mod=args.mod, normalize_abnormal=args.normalize_abnormal)

    if args.type == 'bc':
        if args.abtype is None:
            raise ValueError("For binary classification, --abtype must be specified.")
        generator.generate_binary_class_data(abtype=args.abtype)
        data = generator.get_data()
        labels = generator.get_labels()
        
#     #############################

#         # Visualize original data (before normalization)
#         print("Visualizing original data...")
#         generator.visualize_data()

#         # Visualize normalized data
#         print("Visualizing normalized data...")
#         generator.normalize_abnormal = True
#         generator.generate_binary_class_data(abtype=args.abtype)
#         generator.visualize_data()
        
# ##################################
        save_libsvm_format(data, labels, args.data_path)
        
    # # Check if the abnormal data is normalized (each abnormal sample has unit norm)
    # abnormal_data = data[-args.b:, :-1]  # Exclude the label column
    # norms = np.linalg.norm(abnormal_data, axis=1)
    # print("\nNorms of each abnormal sample (should be close to 1):")
    # print(norms)
    
if __name__ == "__main__":
    main()


# For binary classification:
# python data_generator.py -t bc -d binary_synthetic_data.libsvm -w 48 --t 0.5 -a 20 -b 10 -m 1 --abtype 1 --normalize_abnormal

# For multiclass classification:
# python data_generator.py -t mc -d multiclass_synthetic_data.libsvm -w 48 --t 0.5 -a 20 -b 10 -m 1 --normalize_abnormal
