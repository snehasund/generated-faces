# %%
import torch
torch.cuda.is_available()

# %%
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torchvision.datasets import CelebA, ImageFolder # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!  

# %%
def show_tensor_images(image_tensor, num_images=25, size=(3, 64, 64)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

# %%
def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

# %%
# build the generator class
class Generator(nn.Module):
    
    def __init__(self, z_dim=100, im_dim=12288, hidden_dim=64):
        super(Generator, self).__init__()
        # build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            get_generator_block(hidden_dim * 8, hidden_dim * 16),
            nn.Linear(hidden_dim * 16, im_dim),
            nn.Tanh()  # Use Tanh activation for the last layer to output values in [-1, 1]
        )
    
    def forward(self, noise):
        return self.gen(noise)

# %%
def get_noise(batch_size, z_dim, device='cpu'):
    '''
    Function to generate a batch of noise vectors (z) that will be used as input to the generator.
    
    Parameters:
        batch_size: Size of the batch, a scalar
        z_dim: Dimension of the noise vector (z), a scalar
        device: Device (CPU or GPU) where the tensor will be allocated
    
    Returns:
        A tensor of shape (batch_size, z_dim) containing noise vectors sampled from a normal distribution
    '''
    return torch.randn(batch_size, z_dim, device=device)

# %%
def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network block of the discriminator given input and output dimensions.
    
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
        
    Returns:
        a discriminator neural network layer, with a linear transformation 
        followed by an nn.LeakyReLU activation with negative slope of 0.2
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

# %%


# %%
# build the discriminator class
class Discriminator(nn.Module):
    def __init__(self, im_dim=12288, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, image):
        return self.disc(image)

# %%
# training our model !!

# parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
lr = 0.0002

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize image to 64x64
    transforms.ToTensor(),         # Convert image to PyTorch tensor
    # Add more transformations if needed
])



# %%
# Load dataset
dataset_path = 'data/img_align_celeba'
image_dataset = ImageFolder(root=dataset_path, transform=transform)

# Create DataLoader
dataloader = DataLoader(
    dataset=image_dataset,
    batch_size=batch_size,
    shuffle=True
)

# %%

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# %%
# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_disc_loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    fake_noise = get_noise(num_images, z_dim, device=device) # create noise vector
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach()) # get the discriminator's prediction of the fake image
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred)) # calculate loss
    disc_real_pred = disc(real) # get discriminator's prediction of the real image
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred)) #calculate loss
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

# %%
# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    #### START CODE HERE ####
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    #### END CODE HERE ####
    return gen_loss

# %%
import os
import torch

# Define the checkpoint file path
checkpoint_path = "checkpoint.pth"

# initialize epoch from checkpoint if it exists
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
    gen_opt.load_state_dict(checkpoint['gen_opt'])
    disc_opt.load_state_dict(checkpoint['disc_opt'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True  # Whether the generator should be tested
gen_loss = False
error = False
count = start_epoch

for epoch in range(start_epoch, n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}, count: {count}")
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        ### Update generator ###
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()

        # For testing purposes, to check that your code changes the generator weights
        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
    
    # Save checkpoint at the end of each epoch
    checkpoint = {
        'epoch': epoch,
        'gen': gen.state_dict(),
        'disc': disc.state_dict(),
        'gen_opt': gen_opt.state_dict(),
        'disc_opt': disc_opt.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    count += 1
    print(f"Checkpoint saved at epoch {epoch + 1}")

print("Training complete.")