import torch

# Create a tensor with a different batch size
A = torch.arange(5*2*6*10).view(5, 2, 6, 10)  # Now batch_size=5

print(A[[0,0],[0,1],[0,0]])
# # Pokemon indices for each batch and player
# # Let's say we have indices for all 5 batches
# pokemon_indices = torch.tensor([
#     [3, 5],  # batch 0: player 0's 3rd pokemon, player 1's 5th pokemon
#     [0, 2],  # batch 1: player 0's 0th pokemon, player 1's 2nd pokemon
#     [4, 1],  # batch 2
#     [2, 3],  # batch 3
#     [5, 4]   # batch 4
# ])

# # Get dimensions from the tensors
# batch_size = A.size(0)  # 5
# num_players = A.size(1)  # 2

# # Create batch and player indices dynamically based on A's dimensions
# batch_indices = torch.arange(batch_size, device=A.device).view(-1, 1).expand(-1, num_players)
# print(batch_indices)
# player_indices = torch.arange(num_players, device=A.device).view(1, -1).expand(batch_size, -1)

# # Select the Pokémon embeddings
# selected_pokemon = A[batch_indices, player_indices, pokemon_indices]

# print(selected_pokemon.shape)  # Should be (5, 2, 10)