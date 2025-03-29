"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  AlertCircle, 
  Share2, 
  Database,
  Plus,
  Network,
  Tag,
  Link
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';

// Dynamically import ForceGraphComponent to avoid SSR issues
const ForceGraphComponent = dynamic(() => import('@/components/ForceGraphComponent'), { ssr: false });

// Define interfaces
interface Graph {
  id: string;
  name: string;
  entities: Entity[];
  relationships: Relationship[];
  metadata: Record<string, unknown>;
  document_ids: string[];
  filters?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

interface Entity {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
  chunk_sources: Record<string, number[]>;
}

interface Relationship {
  id: string;
  type: string;
  source_id: string;
  target_id: string;
}

interface GraphSectionProps {
  apiBaseUrl: string;
}

// Map entity types to colors
const entityTypeColors: Record<string, string> = {
  'person': '#4f46e5',   // Indigo
  'organization': '#06b6d4', // Cyan
  'location': '#10b981',  // Emerald
  'date': '#f59e0b',     // Amber
  'concept': '#8b5cf6',   // Violet
  'event': '#ec4899',    // Pink
  'product': '#ef4444',   // Red
  'default': '#6b7280'   // Gray
};

const GraphSection: React.FC<GraphSectionProps> = ({ apiBaseUrl }) => {
  // State variables
  const [graphs, setGraphs] = useState<Graph[]>([]);
  const [selectedGraph, setSelectedGraph] = useState<Graph | null>(null);
  const [graphName, setGraphName] = useState('');
  const [graphDocuments, setGraphDocuments] = useState<string[]>([]);
  const [graphFilters, setGraphFilters] = useState('{}');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('list');
  const [showNodeLabels, setShowNodeLabels] = useState(true);
  const [showLinkLabels, setShowLinkLabels] = useState(true);

  // Refs for graph visualization
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const graphInstance = useRef<{ width: (width: number) => unknown } | null>(null);
  
  // Prepare data for force-graph
  const prepareGraphData = useCallback((graph: Graph | null) => {
    if (!graph) return { nodes: [], links: [] };

    const nodes = graph.entities.map(entity => ({
      id: entity.id,
      label: entity.label,
      type: entity.type,
      properties: entity.properties,
      color: entityTypeColors[entity.type.toLowerCase()] || entityTypeColors.default
    }));

    // Create a Set of all entity IDs for faster lookups
    const nodeIdSet = new Set(graph.entities.map(entity => entity.id));

    // Filter relationships to only include those where both source and target nodes exist
    const links = graph.relationships
      .filter(rel => nodeIdSet.has(rel.source_id) && nodeIdSet.has(rel.target_id))
      .map(rel => ({
        source: rel.source_id,
        target: rel.target_id,
        type: rel.type
      }));

    return { nodes, links };
  }, []);

  // Initialize force-graph visualization
  const initializeGraph = useCallback(() => {
    // No need for implementation as ForceGraphComponent handles this
  }, []);

  // Handle window resize for responsive graph
  useEffect(() => {
    const handleResize = () => {
      if (graphContainerRef.current && graphInstance.current) {
        graphInstance.current.width(graphContainerRef.current.clientWidth);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Fetch all graphs
  const fetchGraphs = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/graphs`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch graphs: ${response.statusText}`);
      }
      
      const data = await response.json();
      setGraphs(data);
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error fetching graphs: ${error.message}`);
      console.error('Error fetching graphs:', err);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  // Fetch graphs on component mount
  useEffect(() => {
    fetchGraphs();
  }, [fetchGraphs]);

  // Fetch a specific graph
  const fetchGraph = async (graphName: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/graph/${encodeURIComponent(graphName)}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch graph: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSelectedGraph(data);
      
      // Change to visualize tab if we're not on the list tab
      if (activeTab !== 'list' && activeTab !== 'create') {
        setActiveTab('visualize');
      }
      
      return data;
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error fetching graph: ${error.message}`);
      console.error('Error fetching graph:', err);
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Handle graph click
  const handleGraphClick = (graph: Graph) => {
    fetchGraph(graph.name);
  };

  // Create a new graph
  const handleCreateGraph = async () => {
    if (!graphName.trim()) {
      setError('Please enter a graph name');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // Parse filters
      let parsedFilters = {};
      try {
        parsedFilters = JSON.parse(graphFilters);
      } catch {
        throw new Error('Invalid JSON in filters field');
      }
      
      const response = await fetch(`${apiBaseUrl}/graph/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: graphName,
          filters: Object.keys(parsedFilters).length > 0 ? parsedFilters : undefined,
          documents: graphDocuments.length > 0 ? graphDocuments : undefined,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to create graph: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSelectedGraph(data);
      
      // Refresh the graphs list
      await fetchGraphs();
      
      // Reset form
      setGraphName('');
      setGraphDocuments([]);
      setGraphFilters('{}');
      
      // Switch to visualize tab
      setActiveTab('visualize');
      
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error creating graph: ${error.message}`);
      console.error('Error creating graph:', err);
    } finally {
      setLoading(false);
    }
  };

  // Initialize or update graph visualization when the selected graph changes
  useEffect(() => {
    if (selectedGraph && activeTab === 'visualize') {
      // Use setTimeout to ensure the container is rendered
      setTimeout(() => {
        initializeGraph();
      }, 100);
    }
  }, [selectedGraph, activeTab, initializeGraph, showNodeLabels, showLinkLabels]);

  // Handle tab change
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    // Initialize graph if switching to visualize tab and a graph is selected
    if (value === 'visualize' && selectedGraph) {
      setTimeout(() => {
        initializeGraph();
      }, 100);
    }
  };

  // Render the graph visualization tab
  const renderVisualization = () => {
    if (!selectedGraph) {
      return (
        <div className="flex items-center justify-center h-[600px] border rounded-md">
          <div className="text-center p-8">
            <Network className="h-16 w-16 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-medium mb-2">No Graph Selected</h3>
            <p className="text-gray-500">
              Select a graph from the list to visualize it here.
            </p>
          </div>
        </div>
      );
    }

    return (
      <div className="border rounded-md">
        <div className="p-4 border-b flex justify-between items-center">
          <div>
            <h3 className="text-lg font-medium">{selectedGraph.name}</h3>
            <p className="text-sm text-gray-500">
              {selectedGraph.entities.length} entities, {selectedGraph.relationships.length} relationships
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Tag className="h-4 w-4" />
            <Label htmlFor="show-node-labels" className="text-sm cursor-pointer">Show Node Labels</Label>
            <Switch 
              id="show-node-labels" 
              checked={showNodeLabels} 
              onCheckedChange={setShowNodeLabels} 
            />
          </div>
          <div className="flex items-center gap-2">
            <Link className="h-4 w-4" />
            <Label htmlFor="show-link-labels" className="text-sm cursor-pointer">Show Relationship Labels</Label>
            <Switch 
              id="show-link-labels" 
              checked={showLinkLabels} 
              onCheckedChange={setShowLinkLabels} 
            />
          </div>
        </div>
        <div ref={graphContainerRef} className="h-[600px]">
          <ForceGraphComponent 
            data={prepareGraphData(selectedGraph)}
            width={graphContainerRef.current?.clientWidth || 800}
            height={600}
            showNodeLabels={showNodeLabels}
            showLinkLabels={showLinkLabels}
          />
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold flex items-center">
          <Network className="mr-2 h-6 w-6" />
          Knowledge Graphs
        </h2>
        {selectedGraph && (
          <div className="flex items-center">
            <Badge variant="outline" className="text-md px-3 py-1 bg-blue-50">
              Current Graph: {selectedGraph.name}
            </Badge>
          </div>
        )}
      </div>

      <Tabs defaultValue="list" value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="mb-4">
          <TabsTrigger value="list">Available Graphs</TabsTrigger>
          <TabsTrigger value="create">Create New Graph</TabsTrigger>
          <TabsTrigger value="visualize" disabled={!selectedGraph}>Visualize Graph</TabsTrigger>
        </TabsList>

        {/* Graph List Tab */}
        <TabsContent value="list">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Database className="mr-2 h-5 w-5" />
                Available Knowledge Graphs
              </CardTitle>
              <CardDescription>
                Select a graph to view its details or visualize it.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex justify-center items-center p-8">
                  <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
                </div>
              ) : graphs.length === 0 ? (
                <div className="text-center p-8 border-2 border-dashed rounded-lg">
                  <Network className="mx-auto h-12 w-12 mb-3 text-gray-400" />
                  <p className="text-gray-500 mb-3">No graphs available.</p>
                  <Button onClick={() => setActiveTab('create')} variant="default">
                    <Plus className="mr-2 h-4 w-4" />
                    Create Your First Graph
                  </Button>
                </div>
              ) : (
                <ScrollArea className="h-[400px] pr-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {graphs.map((graph) => (
                      <Card
                        key={graph.id}
                        className={`cursor-pointer hover:shadow-md transition-shadow ${
                          selectedGraph?.id === graph.id ? 'border-2 border-blue-500' : ''
                        }`}
                        onClick={() => handleGraphClick(graph)}
                      >
                        <CardHeader className="pb-2">
                          <CardTitle className="flex justify-between items-center">
                            <span>{graph.name}</span>
                            <Badge variant="outline">
                              {new Date(graph.created_at).toLocaleDateString()}
                            </Badge>
                          </CardTitle>
                          <CardDescription>
                            {graph.entities.length} entities, {graph.relationships.length} relationships
                          </CardDescription>
                        </CardHeader>
                        <CardContent>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {Array.from(new Set(graph.entities.map(e => e.type))).slice(0, 5).map(type => (
                              <Badge 
                                key={type}
                                style={{ backgroundColor: entityTypeColors[type.toLowerCase()] || entityTypeColors.default }}
                                className="text-white"
                              >
                                {type}
                              </Badge>
                            ))}
                            {Array.from(new Set(graph.entities.map(e => e.type))).length > 5 && (
                              <Badge variant="outline">+{Array.from(new Set(graph.entities.map(e => e.type))).length - 5} more</Badge>
                            )}
                          </div>
                          <div className="mt-3 text-sm text-gray-500">
                            {graph.document_ids.length} document{graph.document_ids.length !== 1 ? 's' : ''}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>

          {selectedGraph && (
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>{selectedGraph.name}</CardTitle>
                <CardDescription>
                  Graph Details and Statistics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                    <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-1">Documents</h4>
                    <div className="text-2xl font-bold">{selectedGraph.document_ids.length}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">source documents</div>
                  </div>
                  
                  <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg">
                    <h4 className="font-medium text-emerald-700 dark:text-emerald-300 mb-1">Entities</h4>
                    <div className="text-2xl font-bold">{selectedGraph.entities.length}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">unique elements</div>
                  </div>
                  
                  <div className="bg-amber-50 dark:bg-amber-900/20 p-4 rounded-lg">
                    <h4 className="font-medium text-amber-700 dark:text-amber-300 mb-1">Relationships</h4>
                    <div className="text-2xl font-bold">{selectedGraph.relationships.length}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">connections</div>
                  </div>
                  
                  <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                    <h4 className="font-medium text-purple-700 dark:text-purple-300 mb-1">Created</h4>
                    <div className="text-xl font-bold">{new Date(selectedGraph.created_at).toLocaleDateString()}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {new Date(selectedGraph.created_at).toLocaleTimeString()}
                    </div>
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium mb-2">Entity Types</h4>
                    <div className="bg-gray-50 p-3 rounded-md">
                      {Object.entries(
                        selectedGraph.entities.reduce((acc, entity) => {
                          acc[entity.type] = (acc[entity.type] || 0) + 1;
                          return acc;
                        }, {} as Record<string, number>)
                      ).map(([type, count]) => (
                        <div key={type} className="flex justify-between mb-2">
                          <div className="flex items-center">
                            <div 
                              className="w-3 h-3 rounded-full mr-2" 
                              style={{ backgroundColor: entityTypeColors[type.toLowerCase()] || entityTypeColors.default }}
                            ></div>
                            <span>{type}</span>
                          </div>
                          <Badge variant="outline">{count}</Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Relationship Types</h4>
                    <div className="bg-gray-50 p-3 rounded-md">
                      {Object.entries(
                        selectedGraph.relationships.reduce((acc, rel) => {
                          acc[rel.type] = (acc[rel.type] || 0) + 1;
                          return acc;
                        }, {} as Record<string, number>)
                      ).map(([type, count]) => (
                        <div key={type} className="flex justify-between mb-2">
                          <span>{type}</span>
                          <Badge variant="outline">{count}</Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="mt-6 flex justify-end">
                  <Button onClick={() => setActiveTab('visualize')}>
                    <Share2 className="mr-2 h-4 w-4" />
                    Visualize Graph
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Create Graph Tab */}
        <TabsContent value="create">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Plus className="mr-2 h-5 w-5" />
                Create New Knowledge Graph
              </CardTitle>
              <CardDescription>
                Create a knowledge graph from documents in your collection to enhance your queries.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="graph-name">Graph Name</Label>
                  <Input
                    id="graph-name"
                    placeholder="Enter a unique name for your graph"
                    value={graphName}
                    onChange={(e) => setGraphName(e.target.value)}
                  />
                  <p className="text-sm text-gray-500">
                    Give your graph a descriptive name that helps you identify its purpose.
                  </p>
                </div>

                <div className="border-t pt-4 mt-4">
                  <h3 className="text-md font-medium mb-3">Document Selection</h3>
                  <p className="text-sm text-gray-500 mb-3">
                    Choose which documents to include in your graph. You can specify document IDs directly or use metadata filters.
                  </p>
                  
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="graph-documents">Document IDs (Optional)</Label>
                      <Textarea
                        id="graph-documents"
                        placeholder="Enter document IDs separated by commas"
                        value={graphDocuments.join(', ')}
                        onChange={(e) => setGraphDocuments(e.target.value.split(',').map(id => id.trim()).filter(id => id))}
                        className="min-h-[80px]"
                      />
                      <p className="text-xs text-gray-500">
                        Specify document IDs to include in the graph, or leave empty and use filters below.
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="graph-filters">Metadata Filters (Optional)</Label>
                      <Textarea
                        id="graph-filters"
                        placeholder='{"category": "research", "author": "Jane Doe"}'
                        value={graphFilters}
                        onChange={(e) => setGraphFilters(e.target.value)}
                        className="min-h-[80px] font-mono"
                      />
                      <p className="text-xs text-gray-500">
                        JSON object with metadata filters to select documents. All documents matching these filters will be included.
                      </p>
                    </div>
                  </div>
                </div>

                <Button 
                  onClick={handleCreateGraph} 
                  disabled={!graphName || loading}
                  className="w-full"
                >
                  {loading ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  ) : null}
                  Create Knowledge Graph
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Visualize Graph Tab */}
        <TabsContent value="visualize">
          {renderVisualization()}
        </TabsContent>
      </Tabs>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default GraphSection;