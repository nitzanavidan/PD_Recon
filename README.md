# PD_Recon: Deep Learning-Based Fast MRI Reconstruction:Improving Generalization for Clinical Translation
Numerous deep neural network (DNN)-based methods have been proposed in recent years to tackle the challenging ill-posed inverse problem of MRI reconstruction from undersampled ’k-space’ (Fourier domain) data. However, these methods have shown instability when faced with variations in the acquisition process and anatomical distribution. This instability indicates that DNN architectures have poorer generalization compared to their classical counterparts in capturing the relevant physical models. Consequently, the limited generalization hinders the applicability of DNNs for undersampled MRI reconstruction in the clinical setting, which is especially critical in detecting subtle pathological regions that play a crucial role in clinical diagnosis. We enhance the generalization capacity of deep neural network (DNN) methods for undersampled MRI reconstruction by introducing a physically-primed DNN architecture and training approach. Our architecture incorporates the undersampling mask into the model and utilizes a specialized training method that leverages data generated with various undersampling masks to encourage the model to generalize the undersampled MRI reconstruction problem. Through extensive experimentation on the publicly available Fast-MRI dataset, we demonstrate the added value of our approach. Our physically-primed approach exhibits significantly improved robustness against variations in the acquisition process and anatomical distribution, particularly in pathological regions, compared to both vanilla DNN methods and DNNs trained with undersampling mask augmentation.