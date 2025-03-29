"use client";

import React, { useEffect, useRef } from 'react';

interface NodeObject {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
  color: string;
  x?: number;
  y?: number;
}

interface LinkObject {
  source: string | NodeObject;
  target: string | NodeObject;
  type: string;
}

interface ForceGraphComponentProps {
  data: {
    nodes: NodeObject[];
    links: LinkObject[];
  };
  width: number;
  height: number;
  showNodeLabels?: boolean;
  showLinkLabels?: boolean;
}

// Define types for the force-graph library
interface ForceGraphInstance {
  width: (width: number) => ForceGraphInstance;
  height: (height: number) => ForceGraphInstance;
  graphData: (data: unknown) => ForceGraphInstance;
  nodeLabel: (callback: (node: NodeObject) => string) => ForceGraphInstance;
  nodeColor: (callback: (node: NodeObject) => string) => ForceGraphInstance;
  linkLabel: (callback: (link: LinkObject) => string) => ForceGraphInstance;
  linkDirectionalArrowLength: (length: number) => ForceGraphInstance;
  linkDirectionalArrowRelPos: (pos: number) => ForceGraphInstance;
  nodeCanvasObject?: (callback: (node: NodeObject, ctx: CanvasRenderingContext2D, globalScale: number) => void) => ForceGraphInstance;
  linkCanvasObject?: (callback: (link: LinkObject, ctx: CanvasRenderingContext2D, globalScale: number) => void) => ForceGraphInstance;
  _destructor?: () => void;
}

const ForceGraphComponent: React.FC<ForceGraphComponentProps> = ({ 
  data, 
  width, 
  height, 
  showNodeLabels = true, 
  showLinkLabels = true 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Clear previous graph
    containerRef.current.innerHTML = '';
    
    let graphInstance: ForceGraphInstance | null = null;
    
    // Dynamically import and initialize the force-graph
    const initGraph = async () => {
      try {
        // Dynamic import
        const ForceGraphModule = await import('force-graph');
        
        // Get the ForceGraph constructor function
        const ForceGraphConstructor = ForceGraphModule.default;
        
        // Create a new graph instance using the 'new' keyword
        if (containerRef.current) {
          graphInstance = new ForceGraphConstructor(containerRef.current) as ForceGraphInstance;
          
          // Configure the graph
          const graph = graphInstance
            .width(width)
            .height(height)
            .graphData({
              nodes: data.nodes.map(node => ({...node})),
              links: data.links.map(link => ({
                source: link.source,
                target: link.target,
                type: link.type
              }))
            })
            .nodeLabel((node: NodeObject) => `${node.label} (${node.type})`)
            .nodeColor((node: NodeObject) => node.color)
            .linkLabel((link: LinkObject) => link.type)
            .linkDirectionalArrowLength(3)
            .linkDirectionalArrowRelPos(1);
          
          // Always use nodeCanvasObject to have consistent rendering regardless of label visibility
          if (graph.nodeCanvasObject) {
            graph.nodeCanvasObject((node: NodeObject, ctx: CanvasRenderingContext2D, globalScale: number) => {
              // Draw the node circle
              const nodeR = 5;
              
              if (typeof node.x !== 'number' || typeof node.y !== 'number') return;
              
              const x = node.x;
              const y = node.y;
              
              ctx.beginPath();
              ctx.arc(x, y, nodeR, 0, 2 * Math.PI);
              ctx.fillStyle = node.color;
              ctx.fill();
              
              // Only draw the text label if showNodeLabels is true
              if (showNodeLabels) {
                const label = node.label;
                const fontSize = 12/globalScale;
                
                ctx.font = `${fontSize}px Sans-Serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                // Add a background for better readability
                const textWidth = ctx.measureText(label).width;
                const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);
                
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.fillRect(
                  x - bckgDimensions[0] / 2,
                  y - bckgDimensions[1] / 2,
                  bckgDimensions[0],
                  bckgDimensions[1]
                );
                
                ctx.fillStyle = 'black';
                ctx.fillText(label, x, y);
              }
            });
          }
          
          // Always use linkCanvasObject for consistent rendering
          if (graph.linkCanvasObject) {
            graph.linkCanvasObject((link: LinkObject, ctx: CanvasRenderingContext2D, globalScale: number) => {
              // Draw the link line
              const start = link.source as NodeObject;
              const end = link.target as NodeObject;
              
              if (!start || !end || typeof start.x !== 'number' || typeof end.x !== 'number' || 
                  typeof start.y !== 'number' || typeof end.y !== 'number') return;
              
              const startX = start.x;
              const startY = start.y;
              const endX = end.x;
              const endY = end.y;
              
              ctx.beginPath();
              ctx.moveTo(startX, startY);
              ctx.lineTo(endX, endY);
              ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
              ctx.lineWidth = 1;
              ctx.stroke();
              
              // Draw arrowhead regardless of label visibility
              const arrowLength = 5;
              const dx = endX - startX;
              const dy = endY - startY;
              const angle = Math.atan2(dy, dx);
              
              // Calculate a position near the target for the arrow
              const arrowDistance = 15; // Distance from target node
              const arrowX = endX - Math.cos(angle) * arrowDistance;
              const arrowY = endY - Math.sin(angle) * arrowDistance;
              
              ctx.beginPath();
              ctx.moveTo(arrowX, arrowY);
              ctx.lineTo(
                arrowX - arrowLength * Math.cos(angle - Math.PI / 6),
                arrowY - arrowLength * Math.sin(angle - Math.PI / 6)
              );
              ctx.lineTo(
                arrowX - arrowLength * Math.cos(angle + Math.PI / 6),
                arrowY - arrowLength * Math.sin(angle + Math.PI / 6)
              );
              ctx.closePath();
              ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
              ctx.fill();
              
              // Only draw label if showLinkLabels is true
              if (showLinkLabels) {
                const label = link.type;
                if (label) {
                  const fontSize = 10/globalScale;
                  ctx.font = `${fontSize}px Sans-Serif`;
                  
                  // Calculate middle point
                  const middleX = startX + (endX - startX) / 2;
                  const middleY = startY + (endY - startY) / 2;
                  
                  // Add a background for better readability
                  const textWidth = ctx.measureText(label).width;
                  const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);
                  
                  ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                  ctx.fillRect(
                    middleX - bckgDimensions[0] / 2,
                    middleY - bckgDimensions[1] / 2,
                    bckgDimensions[0],
                    bckgDimensions[1]
                  );
                  
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillStyle = 'black';
                  ctx.fillText(label, middleX, middleY);
                }
              }
            });
          }
        }
      } catch (error) {
        console.error("Error initializing force graph:", error);
        
        // Show error message if graph initialization fails
        if (containerRef.current) {
          containerRef.current.innerHTML = `
            <div class="flex items-center justify-center h-full">
              <div class="text-center p-4">
                <h3 class="text-lg font-medium mb-2">Graph Visualization Error</h3>
                <p class="text-sm text-gray-500">
                  There was an error initializing the graph visualization.
                </p>
              </div>
            </div>
          `;
        }
      }
    };
    
    initGraph();
    
    // Cleanup function
    return () => {
      if (graphInstance && typeof graphInstance._destructor === 'function') {
        graphInstance._destructor();
      }
    };
  }, [data, width, height, showNodeLabels, showLinkLabels]);
  
  return <div ref={containerRef} className="w-full h-full" />;
};

export default ForceGraphComponent; 