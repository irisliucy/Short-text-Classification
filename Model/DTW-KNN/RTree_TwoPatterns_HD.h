/**
 * @file RTree.h
 * Definition of a R-tree class which can be used as a generic dictionary
 * (insert-only).
 *
 * @author LUO Lintong
 * @date Spring 2016
 */

#ifndef RTREE_H
#define RTREE_H

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
using namespace std;

#define M 5000 //The number of time series
#define T 128  //The length of a time serie
#define D 24 //The dimension of a time point
#define bandwidth  0.12*T//Used for Sakoe-Chiba Band restriction, 0<=bandwidth<=T
//#define slope_variance 1 //Used for Itakura parallelogram restriction
#define constraint 4 //LB_Keogh window, must be smaller than T
#define PAAReductionFactor 4 //the equal amount for each cell, must be a factor of T
#define BlockNum T/PAAReductionFactor

#define MAXNODES 32
#define MINNODES 16
/**
 * RTree class. Provides interfaces for inserting and finding elements in
 * R-tree.
 */
class RTree
{
public:
    typedef bool (*t_resultCallback)(int, void*);
    
    /// Minimal bounding rectangle (D-dimensional)
    struct Rect
    {
        float m_min[BlockNum][D];                      ///< Min length of bounding box
        float m_max[BlockNum][D];                      ///< Max length of bounding box
    };
    /// May be data or may be another subtree
    /// The parents level determines this.
    /// If the parents level is 0, then this is data
    /// Node for each branch level
    struct Node;
    struct Branch
    {
        Rect m_rect;                                  ///< Bounds
        Node* m_child;                                ///< Child node
        int m_data;                              ///< Data Id
    };
    struct Node
    {
        bool IsInternalNode(){ return (m_level > 0); }; // Not a leaf, but a internal node
        bool IsLeaf(){ return (m_level == 0); }; // A leaf, contains data
        
        int m_count;                                  ///< Count
        int m_level;                                  ///< Leaf is zero, others positive
        Branch m_branch[MAXNODES];                    ///< Branch
    };


        /// Variables for finding a split partition
    struct PartitionVars
    {
        enum { NOT_TAKEN = -1 }; // indicates that position
        
        int m_partition[MAXNODES+1];
        int m_total;
        int m_minFill;
        int m_count[2];
        Rect m_cover[2];
        float m_area[2];
        
        Branch m_branchBuf[MAXNODES+1];
        int m_branchCount;
        Rect m_coverSplit;
        float m_coverSplitArea;
    };
    Node* m_root;
    //float m_unitRecArea;
    
    //Constructor
    RTree()
    {

        m_root = AllocNode();
        m_root->m_level = 0;
        //m_unitSphereVolume = (ELEMTYPEREAL)UNIT_SPHERE_VOLUMES[D];
    }
    void InitNode(Node* a_node)
    {
        a_node->m_count = 0;
        a_node->m_level = -1;
    }
    Node* AllocNode()
    {
        Node* newNode;
        newNode = new Node;
        InitNode(newNode);
        return newNode;
    }

    
    //Destructor
    void FreeNode(Node* a_node)
    {
        delete a_node;
    }
    void RemoveAllRec(Node* a_node)
    {
        
        if(a_node->IsInternalNode()) // This is an internal node in the tree
        {
            for(int index=0; index < a_node->m_count; ++index)
            {
                RemoveAllRec(a_node->m_branch[index].m_child);
            }
        }
        FreeNode(a_node);
    }
    void Reset()
    {
        RemoveAllRec(m_root);// Delete all existing nodes
    }
    ~RTree()
    {
        Reset(); // Free, or reset node memory
    }

    

    float CalcDiff(Rect* a_rect)
    {
        float diff=0;
        for(int d=0;d<D;d++){
            for(int index=0;index<BlockNum;index++){
                diff+=a_rect->m_max[index][d] - a_rect->m_min[index][d];
            }
        }
        return diff;
    }
    // Combine two rectangles into larger one containing both
    Rect CombineRect(const Rect* a_rectA, const Rect* a_rectB)
    {
        Rect newRect;
        
        for(int d=0;d<D;d++){
            for(int index = 0; index < BlockNum; ++index)
            {
                newRect.m_min[index][d] = min(a_rectA->m_min[index][d], a_rectB->m_min[index][d]);
                newRect.m_max[index][d] = max(a_rectA->m_max[index][d], a_rectB->m_max[index][d]);
            }
        }
        
        return newRect;
    }
    // Pick a branch.  Pick the one that will need the smallest increase
    // in area to accomodate the new rectangle.  This will result in the
    // least total area for the covering rectangles in the current node.
    // In case of a tie, pick the one which was smaller before, to get
    // the best resolution when searching.
    int PickBranch(const Rect* a_rect, Node* a_node)
    {
        
        bool firstTime = true;
        float increase;
        float bestIncr = (float)-1;
        float area;
        float bestArea;
        int best;
        Rect tempRect;
        
        for(int index=0; index < a_node->m_count; ++index)
        {
            Rect* curRect = &a_node->m_branch[index].m_rect;
            area = CalcDiff(curRect);
            tempRect = CombineRect(a_rect, curRect);
            increase = CalcDiff(&tempRect) - area;
            if((increase < bestIncr) || firstTime)
            {
                best = index;
                bestArea = area;
                bestIncr = increase;
                firstTime = false;
            }
            else if((increase == bestIncr) && (area < bestArea))
            {
                best = index;
                bestArea = area;
                bestIncr = increase;
            }
        }
        return best;
    }
    // Find the smallest rectangle that includes all rectangles in branches of a node.
    Rect NodeCover(Node* a_node)
    {
        Rect rect = a_node->m_branch[0].m_rect;
        for(int index = 1; index < a_node->m_count; ++index)
        {
            rect = CombineRect(&rect, &(a_node->m_branch[index].m_rect));
        }
        return rect;
    }
    
    void InitParVars(PartitionVars* a_parVars, int a_maxRects, int a_minFill)
    {
        assert(a_parVars);
        
        a_parVars->m_count[0] = a_parVars->m_count[1] = 0;
        a_parVars->m_area[0] = a_parVars->m_area[1] = (float)0;
        a_parVars->m_total = a_maxRects;
        a_parVars->m_minFill = a_minFill;
        for(int index=0; index < a_maxRects; ++index)
        {
            a_parVars->m_partition[index] = PartitionVars::NOT_TAKEN;
        }
    }
    void GetBranches(Node* a_node, const Branch* a_branch, PartitionVars* a_parVars)
    {
        
        // Load the branch buffer
        for(int index=0; index < MAXNODES; ++index)
        {
            a_parVars->m_branchBuf[index] = a_node->m_branch[index];
        }
        a_parVars->m_branchBuf[MAXNODES] = *a_branch;
        a_parVars->m_branchCount = MAXNODES + 1;
        
        // Calculate rect containing all in the set
        a_parVars->m_coverSplit = a_parVars->m_branchBuf[0].m_rect;
        for(int index=1; index < MAXNODES+1; ++index)
        {
            a_parVars->m_coverSplit = CombineRect(&a_parVars->m_coverSplit, &a_parVars->m_branchBuf[index].m_rect);
        }
        a_parVars->m_coverSplitArea = CalcDiff(&a_parVars->m_coverSplit);
    }
    
    void Classify(int a_index, int a_group, PartitionVars* a_parVars)
    {
        assert(a_parVars);
        assert(PartitionVars::NOT_TAKEN == a_parVars->m_partition[a_index]);
        
        a_parVars->m_partition[a_index] = a_group;
        
        // Calculate combined rect
        if (a_parVars->m_count[a_group] == 0)
        {
            a_parVars->m_cover[a_group] = a_parVars->m_branchBuf[a_index].m_rect;
        }
        else
        {
            a_parVars->m_cover[a_group] = CombineRect(&a_parVars->m_branchBuf[a_index].m_rect, &a_parVars->m_cover[a_group]);
        }
        
        // Calculate volume of combined rect
        a_parVars->m_area[a_group] = CalcDiff(&a_parVars->m_cover[a_group]);
        
        ++a_parVars->m_count[a_group];
    }

    void PickSeeds(PartitionVars* a_parVars)
    {
        int seed0, seed1;
        float worst, waste;
        float area[MAXNODES+1];
        
        for(int index=0; index<a_parVars->m_total; ++index)
        {
            area[index] = CalcDiff(&a_parVars->m_branchBuf[index].m_rect);
        }
        
        worst = -a_parVars->m_coverSplitArea - 1;
        for(int indexA=0; indexA < a_parVars->m_total-1; ++indexA)
        {
            for(int indexB = indexA+1; indexB < a_parVars->m_total; ++indexB)
            {
                Rect oneRect = CombineRect(&a_parVars->m_branchBuf[indexA].m_rect, &a_parVars->m_branchBuf[indexB].m_rect);
                waste = CalcDiff(&oneRect) - area[indexA] - area[indexB];
                if(waste > worst)
                {
                    worst = waste;
                    seed0 = indexA;
                    seed1 = indexB;
                }
            }
        }
        
        Classify(seed0, 0, a_parVars);
        Classify(seed1, 1, a_parVars);
    }
    void ChoosePartition(PartitionVars* a_parVars, int a_minFill)
    {
        assert(a_parVars);
        
        float biggestDiff;
        int group, chosen, betterGroup;
        
        InitParVars(a_parVars, a_parVars->m_branchCount, a_minFill);
        PickSeeds(a_parVars);
        
        while (((a_parVars->m_count[0] + a_parVars->m_count[1]) < a_parVars->m_total)
               && (a_parVars->m_count[0] < (a_parVars->m_total - a_parVars->m_minFill))
               && (a_parVars->m_count[1] < (a_parVars->m_total - a_parVars->m_minFill)))
        {
            biggestDiff = (float) -1;
            for(int index=0; index<a_parVars->m_total; ++index)
            {
                if(PartitionVars::NOT_TAKEN == a_parVars->m_partition[index])
                {
                    Rect* curRect = &a_parVars->m_branchBuf[index].m_rect;
                    Rect rect0 = CombineRect(curRect, &a_parVars->m_cover[0]);
                    Rect rect1 = CombineRect(curRect, &a_parVars->m_cover[1]);
                    float growth0 = CalcDiff(&rect0) - a_parVars->m_area[0];
                    float growth1 = CalcDiff(&rect1) - a_parVars->m_area[1];
                    float diff = growth1 - growth0;
                    if(diff >= 0)
                    {
                        group = 0;
                    }
                    else
                    {
                        group = 1;
                        diff = -diff;
                    }
                    
                    if(diff > biggestDiff)
                    {
                        biggestDiff = diff;
                        chosen = index;
                        betterGroup = group;
                    }
                    else if((diff == biggestDiff) && (a_parVars->m_count[group] < a_parVars->m_count[betterGroup]))
                    {
                        chosen = index;
                        betterGroup = group;
                    }
                }
            }
            Classify(chosen, betterGroup, a_parVars);
        }
        
        // If one group too full, put remaining rects in the other
        if((a_parVars->m_count[0] + a_parVars->m_count[1]) < a_parVars->m_total)
        {
            if(a_parVars->m_count[0] >= a_parVars->m_total - a_parVars->m_minFill)
            {
                group = 1;
            }
            else
            {
                group = 0;
            }
            for(int index=0; index<a_parVars->m_total; ++index)
            {
                if(PartitionVars::NOT_TAKEN == a_parVars->m_partition[index])
                {
                    Classify(index, group, a_parVars);
                }
            }
        }
        
        assert((a_parVars->m_count[0] + a_parVars->m_count[1]) == a_parVars->m_total);
        assert((a_parVars->m_count[0] >= a_parVars->m_minFill) &&
               (a_parVars->m_count[1] >= a_parVars->m_minFill));
    }
    void LoadNodes(Node* a_nodeA, Node* a_nodeB, PartitionVars* a_parVars)
    {
        for(int index=0; index < a_parVars->m_total; ++index)
        {
            assert(a_parVars->m_partition[index] == 0 || a_parVars->m_partition[index] == 1);
            
            int targetNodeIndex = a_parVars->m_partition[index];
            Node* targetNodes[] = {a_nodeA, a_nodeB};
            
            // It is assured that AddBranch here will not cause a node split.
            bool nodeWasSplit = AddBranch(&a_parVars->m_branchBuf[index], targetNodes[targetNodeIndex], NULL);
            assert(!nodeWasSplit);
        }
    }
    void SplitNode(Node* a_node, const Branch* a_branch, Node** a_newNode)
    {
        assert(a_node);
        assert(a_branch);
        
        // Could just use local here, but member or external is faster since it is reused
        PartitionVars localVars;
        PartitionVars* parVars = &localVars;
        
        // Load all the branches into a buffer, initialize old node
        GetBranches(a_node, a_branch, parVars);
        
        // Find partition
        ChoosePartition(parVars, MINNODES);
        
        // Create a new node to hold (about) half of the branches
        *a_newNode = AllocNode();
        (*a_newNode)->m_level = a_node->m_level;
        
        // Put branches from buffer into 2 nodes according to the chosen partition
        a_node->m_count = 0;
        LoadNodes(a_node, *a_newNode, parVars);
        
        assert((a_node->m_count + (*a_newNode)->m_count) == parVars->m_total);
    }

    bool AddBranch(const Branch* a_branch, Node* a_node, Node** a_newNode)
    {
        
        if(a_node->m_count < MAXNODES)  // Split won't be necessary
        {
            a_node->m_branch[a_node->m_count] = *a_branch;
            ++a_node->m_count;
            
            return false;
        }
        else
        {
            SplitNode(a_node, a_branch, a_newNode);
            return true;
        }
    }
    // Inserts a new data rectangle into the index structure.
    // Recursively descends tree, propagates splits back up.
    // Returns 0 if node was not split.  Old node updated.
    // If node was split, returns 1 and sets the pointer pointed to by
    // new_node to point to the new node.  Old node updated to become one of two.
    // The level argument specifies the number of steps up from the leaf
    // level to insert; e.g. a data rectangle goes in at level = 0.
    bool InsertRectRec(const Branch& a_branch, Node* a_node, Node** a_newNode, int a_level)
    {
        // recurse until we reach the correct level for the new record. data records
        // will always be called with a_level == 0 (leaf)
        if(a_node->m_level > a_level)
        {
            // Still above level for insertion, go down tree recursively
            Node* otherNode;
            
            // find the optimal branch for this record
            int index = PickBranch(&a_branch.m_rect, a_node);
            
            // recursively insert this record into the picked branch
            bool childWasSplit = InsertRectRec(a_branch, a_node->m_branch[index].m_child, &otherNode, a_level);
            
            if (!childWasSplit)
            {
                // Child was not split. Merge the bounding box of the new record with the
                // existing bounding box
                a_node->m_branch[index].m_rect = CombineRect(&a_branch.m_rect, &(a_node->m_branch[index].m_rect));
                return false;
            }
            else
            {
                // Child was split. The old branches are now re-partitioned to two nodes
                // so we have to re-calculate the bounding boxes of each node
                a_node->m_branch[index].m_rect = NodeCover(a_node->m_branch[index].m_child);
                Branch branch;
                branch.m_child = otherNode;
                branch.m_rect = NodeCover(otherNode);
                
                // The old node is already a child of a_node. Now add the newly-created
                // node to a_node as well. a_node might be split because of that.
                return AddBranch(&branch, a_node, a_newNode);
            }
        }
        else if(a_node->m_level == a_level)
        {
            // We have reached level for insertion. Add rect, split if necessary
            return AddBranch(&a_branch, a_node, a_newNode);
        }
        else
        {
            // Should never occur
            assert(0);
            return false;
        }
    }
    bool InsertRect(const Branch& a_branch, Node** a_root, int a_level)
    {
        Node* newNode;
        
        if(InsertRectRec(a_branch, *a_root, &newNode, a_level))  // Root split
        {
            // Grow tree taller and new root
            Node* newRoot = AllocNode();
            newRoot->m_level = (*a_root)->m_level + 1;
            
            Branch branch;
            
            // add old root node as a child of the new root
            branch.m_rect = NodeCover(*a_root);
            branch.m_child = *a_root;
            AddBranch(&branch, newRoot, NULL);
            
            // add the split node as a child of the new root
            branch.m_rect = NodeCover(newNode);
            branch.m_child = newNode;
            AddBranch(&branch, newRoot, NULL);
            
            // set the new root as the root node
            *a_root = newRoot;
            
            return true;
        }
        
        return false;
    }

    void Insert(float** &a_max, float** &a_min, const int& a_dataId)
    {

        
        Branch branch;
        branch.m_data = a_dataId;
        branch.m_child = NULL;
        for(int d=0;d<D;d++){
            for(int axis=0; axis<BlockNum; ++axis)
            {
                branch.m_rect.m_min[axis][d] = a_min[axis][d];
                branch.m_rect.m_max[axis][d] = a_max[axis][d];
            }
        }
        
        InsertRect(branch, &m_root, 0);
    }
    
    bool Overlap(Rect* a_rectA, Rect* a_rectB)
    {
        for(int d=0;d<D;d++){
            for(int index=0; index < BlockNum; ++index)
            {
                if (a_rectA->m_min[index][d] > a_rectB->m_max[index][d] || a_rectB->m_min[index][d] > a_rectA->m_max[index][d]){
                    return false;
                }
            }
        }
        return true;
    }
    
};

#endif /* RTREE_H */
