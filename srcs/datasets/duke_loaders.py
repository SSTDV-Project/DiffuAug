from DiffusionForBreastMRI.dataset_class.duke_dataset import DukeDataset

def get_duke_dataloader(png_dir, train_batchsize=32, img_size=256, num_workers=8):
    img_transform = transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # NOTE: Scale between [-1, 1]
        ]
    )
    
    dataset = DukeDataset(png_dir, img_transform)
    print(len(dataset))
    
    train_lodaer = DataLoader(dataset, batch_size=train_batchsize, shuffle=True, num_workers=num_workers)
    
    return train_lodaer