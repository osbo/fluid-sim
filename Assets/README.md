# Fluid Simulation with Octree Construction

This project implements a GPU-accelerated fluid simulation using hierarchical octree construction from particle data. The simulation builds a hierarchical octree grid from a list of particles in a parallel, bottom-up fashion.

## Overview

The octree construction process groups particles that are close to each other in 3D space using Morton codes, then builds a hierarchical structure that can be used for efficient neighbor finding and collision detection in fluid simulations.

## Input Buffers

- **`particlesBuffer`**: Contains all particle data (position, velocity, etc.)
- **`mortonCodesBuffer`**: Morton code for each particle
- **`particleIndicesBuffer`**: Original index for each particle (0 to N-1)

## Algorithm

### Step 1: Sort Particles

Sort particles based on their Morton codes. This groups particles that are close to each other in 3D space. A radix sort is used for efficiency on the GPU.

```pseudocode
sortedMortonCodes, sortedParticleIndices = RadixSort(mortonCodesBuffer, particleIndicesBuffer)
// Result: sortedMortonCodes and sortedParticleIndices now contain the Morton
// codes and original particle indices, sorted identically.
```

### Step 2: Identify Unique Particles & Create Leaf Nodes

Find the first occurrence of each unique Morton code in the sorted list. Each unique code corresponds to a leaf node in the octree. This is done using a parallel scan (prefix sum) operation.

#### 2a. Mark Unique Particles

Create an "indicators" buffer of the same size as the particle count. For each particle in the sorted list, mark it with a '1' if its Morton code is different from the previous one, otherwise mark with '0'. The first particle is always marked '1'.

```pseudocode
FOR EACH particle i IN PARALLEL:
    IF i == 0 THEN
        indicators[i] = 1
    ELSE IF sortedMortonCodes[i] != sortedMortonCodes[i-1] THEN
        indicators[i] = 1
    ELSE
        indicators[i] = 0
    END IF
END FOR
```

#### 2b. Perform Exclusive Scan (Prefix Sum) on Indicators

This calculates the destination index for each unique particle.

```pseudocode
prefixSums = ExclusiveScan(indicators)
// Result: prefixSums[i] now holds the count of unique particles before index i.
```

#### 2c. Get Total Unique Count

The total number of unique particles (leaf nodes) is the sum of the last element of the prefix sum and the last indicator.

```pseudocode
numNodes = prefixSums[numParticles - 1] + indicators[numParticles - 1]
```

#### 2d. Scatter Unique Indices

Create a new buffer `uniqueIndices` to store the indices of the unique particles. Each thread checks its indicator. If it's 1, it writes its own index into the `uniqueIndices` buffer at the location calculated by the prefix sum.

```pseudocode
FOR EACH particle i IN PARALLEL:
    IF indicators[i] == 1 THEN
        destinationIndex = prefixSums[i]
        uniqueIndices[destinationIndex] = i // Store the index from the *sorted* array
    END IF
END FOR
```

#### 2e. Create Leaf Nodes

For each unique particle, create a leaf node. The node's properties (position, velocity) are the weighted average of all particles sharing that same Morton code. All leaf nodes are initially marked as "active".

```pseudocode
FOR EACH unique particle j IN PARALLEL (from 0 to numNodes-1):
    startIndex = uniqueIndices[j]
    endIndex = (j + 1 < numNodes) ? uniqueIndices[j + 1] : numParticles

    // Aggregate data from all particles in the range [startIndex, endIndex)
    Aggregate particle data from sortedParticleIndices[startIndex...endIndex-1]
    
    // Create the node
    nodesBuffer[j].position = averagedPosition
    nodesBuffer[j].velocities = averagedVelocities
    nodesBuffer[j].mortonCode = sortedMortonCodes[startIndex]
    nodesBuffer[j].layer = 0
    nodeFlagsBuffer[j] = 1 // Mark as active
END FOR
```

### Step 3: Hierarchical Coarsening (Bottom-Up Octree Build)

Iteratively build the octree from the leaf nodes up to the root. In each iteration (layer), we group active nodes from the layer below into parent nodes.

```pseudocode
FOR layer = 1 TO 10:

    // 3a. Find all currently active nodes
    // (Similar to steps 2a-2d, but performed on the nodeFlagsBuffer)
    activeIndicators = MarkActive(nodeFlagsBuffer)
    activePrefixSums = ExclusiveScan(activeIndicators)
    numActiveNodes = activePrefixSums[numNodes - 1] + activeIndicators[numNodes - 1]
    activeIndices = Scatter(activeIndicators, activePrefixSums) // Stores indices of active nodes
    
    IF numActiveNodes == 0 THEN
        BREAK LOOP
    END IF

    // 3b. Identify Unique Parent Nodes among the active nodes
    // Group active nodes by a truncated version of their Morton code.
    // The number of bits to truncate depends on the current layer.
    // (This is again a Mark -> Scan -> Scatter -> Count process)
    prefixBits = 3 * layer

    // Mark nodes with a unique Morton code prefix
    FOR EACH active node i IN PARALLEL (from 0 to numActiveNodes-1):
        currentNodeIndex = activeIndices[i]
        previousNodeIndex = activeIndices[i-1]
        
        isUniquePrefix = (nodeMortonCodes[currentNodeIndex] >> prefixBits) != (nodeMortonCodes[previousNodeIndex] >> prefixBits)
        uniquePrefixIndicators[i] = isUniquePrefix ? 1 : 0
    END FOR
    
    uniquePrefixSums = ExclusiveScan(uniquePrefixIndicators)
    numUniqueActiveNodes = uniquePrefixSums[numActiveNodes-1] + uniquePrefixIndicators[numActiveNodes-1]
    uniqueActiveIndices = Scatter(uniquePrefixIndicators, uniquePrefixSums) // Stores indices into the *activeIndices* array

    // 3c. Process Nodes (Coarsen or Refine)
    // For each group of nodes that share a parent, decide whether to merge them
    // (coarsen) into a single parent node or keep them separate.
    FOR EACH unique parent group k IN PARALLEL (from 0 to numUniqueActiveNodes-1):
        
        // Get the range of active nodes belonging to this parent group
        startIndex_in_active_array = uniqueActiveIndices[k]
        endIndex_in_active_array = (k + 1 < numUniqueActiveNodes) ? uniqueActiveIndices[k + 1] : numActiveNodes
        
        // Check if all nodes in this group are at a finer layer than the current target layer
        canCoarsen = TRUE
        FOR i from startIndex_in_active_array TO endIndex_in_active_array-1:
            nodeIndex = activeIndices[i]
            IF nodesBuffer[nodeIndex].layer <= layer THEN
                canCoarsen = FALSE
                BREAK
            END IF
        END FOR
        
        IF canCoarsen THEN
            // Aggregate all nodes in the group into the first node
            firstNodeIndex = activeIndices[startIndex_in_active_array]
            Aggregate properties of all nodes in the group into nodesBuffer[firstNodeIndex]
            
            // Deactivate the other nodes in the group
            FOR i from startIndex_in_active_array + 1 TO endIndex_in_active_array-1:
                nodeIndexToDeactivate = activeIndices[i]
                nodeFlagsBuffer[nodeIndexToDeactivate] = 0 // Mark as inactive
            END FOR
        ELSE
            // Refine: if any node is coarser than the target layer, bring it down.
            // This step was in the original code to handle specific refinement cases.
            FOR i from startIndex_in_active_array TO endIndex_in_active_array-1:
                nodeIndex = activeIndices[i]
                IF nodesBuffer[nodeIndex].layer > layer THEN
                    nodesBuffer[nodeIndex].layer = layer - 1
                END IF
            END FOR
        END IF
    END FOR

END FOR // End of layer loop
```

## Files

- **`FluidSimulator.cs`**: Main Unity script for the fluid simulation
- **`Nodes.compute`**: GPU compute shader for octree node operations
- **`NodesPrefixSums.compute`**: GPU compute shader for prefix sum operations
- **`Particles.compute`**: GPU compute shader for particle operations
- **`RadixSort.compute`**: GPU compute shader for radix sorting

## Requirements

- Unity 2022.3 or later
- Universal Render Pipeline (URP)
- Compute shader support

## Usage

1. Open the project in Unity
2. Load the `SampleScene` scene
3. The fluid simulation will start automatically
4. Adjust simulation parameters in the `FluidSimulator` component

## License

This project is part of MIT UROP research work.