import matplotlib.pyplot as plt

# Read the data from the file
b_values = []
with open('a.txt', 'r') as file:
    for line in file:
        a, b = line.strip().split()  # Split each line into A and B values
        b_values.append(float(b))    # Convert B to float and store it

# Create x-axis values (indices for the B values)
x_values = range(len(b_values))

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, b_values, marker='o')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('B Values')
plt.title('Line Graph of B Values')

# Add grid for better readability
plt.grid(True)

# Display the plot
plt.savefig('line_graph.png', dpi=300, bbox_inches='tight')
plt.close()
