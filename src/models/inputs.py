import torch


def get_coordinate_grid(h: int, w: int, device: torch.device):
    """
    Create a 2D grid of coordinates.

    Args:
        h (int): The height of the grid.
        w (int): The width of the grid.
        device (torch.device): The device on which to create the grid.

    Returns:
        torch.Tensor: A tensor containing the coordinates of the grid.
    """

    # Create a 1D tensor of evenly spaced values from 0 to 1 with length w
    x = torch.linspace(0, 1, w, device=device)

    # Create a 1D tensor of evenly spaced values from 0 to 1 with length h
    y = torch.linspace(0, 1, h, device=device)

    # Create a 2D grid of coordinates using meshgrid
    xx, yy = torch.meshgrid(x, y, indexing='xy')

    # Stack the flattened coordinate grids and move to the specified device
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    # coords = coords.reshape(h,w,2).permute(2,0,1).unsqueeze(0)
    return coords

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

# 位置编码
def positional_encoding(coords, num_frequencies=10):
    encoded = [coords]
    for i in range(num_frequencies):
        for fn in [torch.sin, torch.cos]:
            encoded.append(fn(2.0 ** i * coords))
    return torch.cat(encoded, dim=-1)