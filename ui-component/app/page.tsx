'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Card } from '@/components/ui/card';
import { ChevronLeft, ChevronRight, Plus, Send, Upload, X, CheckCircle2 } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

interface MetadataValue {
  [key: string]: string | number | boolean | null | undefined | MetadataValue | MetadataValue[];
}

interface Source {
  id: string;
  name: string;
  type: string;
  uploadedAt: Date;
  metadata?: MetadataValue;
  filename?: string;
}

interface DocumentResponse {
  external_id: string;
  content_type: string;
  filename: string;
  metadata: MetadataValue;
  system_metadata: {
    created_at: string;
    updated_at: string;
    version: number;
    content: string;
  };
  storage_info: {
    bucket: string;
    key: string;
  };
}

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

interface ConnectionStatus {
  isConnected: boolean;
  error?: string;
}

interface ParsedUri {
  serverUrl: string;
  token: string;
}

interface Filter {
  key: string;
  value: string;
}

export default function HomePage() {
  const [sources, setSources] = useState<Source[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [message, setMessage] = useState('');
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);
  const [title, setTitle] = useState('My Knowledge Base');
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [panelWidth, setPanelWidth] = useState(384); // 24rem (w-96) in pixels
  const isDragging = useRef(false);
  const lastX = useRef(0);
  const [uri, setUri] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({ isConnected: false });
  const [parsedUri, setParsedUri] = useState<ParsedUri | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [filters, setFilters] = useState<Filter[]>([]);
  const [isFilterModalOpen, setIsFilterModalOpen] = useState(false);
  const [newFilter, setNewFilter] = useState<Filter>({ key: '', value: '' });
  const [uploadMetadata, setUploadMetadata] = useState<Filter[]>([]);
  const [newUploadMetadata, setNewUploadMetadata] = useState<Filter>({ key: '', value: '' });
  const [textContent, setTextContent] = useState('');

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    lastX.current = e.clientX;
    document.body.style.cursor = 'ew-resize';
    // Prevent text selection while dragging
    e.preventDefault();
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging.current) return;

    const delta = e.clientX - lastX.current;
    setPanelWidth(width => {
      const newWidth = Math.max(320, Math.min(800, width + delta)); // Min: 320px, Max: 800px
      return newWidth;
    });
    lastX.current = e.clientX;
  }, []);

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
    document.body.style.cursor = '';
  }, []);

  // Add and remove event listeners
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  const validateAndParseUri = (input: string): ParsedUri | null => {
    try {
      // Strip any protocol prefix
      const withoutProtocol = input.replace(/^.*?:\/\//, '');
      
      // Split into auth and host parts
      const parts = withoutProtocol.split('@');
      if (parts.length !== 2) {
        throw new Error('Missing @ separator');
      }

      const [authPart, hostPort] = parts;
      const [, token] = authPart.split(':');
      if (!token) {
        throw new Error('No token found');
      }

      // Basic JWT format validation (should have 3 parts separated by dots)
      const tokenParts = token.split('.');
      if (tokenParts.length !== 3) {
        throw new Error('Invalid token format - expected JWT token');
      }

      // Construct server URL with http://
      const serverUrl = `http://${hostPort}`;

      return {
        serverUrl,
        token
      };
    } catch (error) {
      console.error('URI parse error:', error);
      return null;
    }
  };

  const makeRequest = useCallback(async (endpoint: string, options: RequestInit = {}) => {
    if (!parsedUri) throw new Error('No URI parsed');
    
    const url = `${parsedUri.serverUrl}${endpoint}`;
    const headers = {
      'Authorization': `Bearer ${parsedUri.token}`,
      ...options.headers
    };

    console.log('Making request to:', url);

    const response = await fetch(url, {
      ...options,
      headers
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => response.statusText);
      throw new Error(`Request failed (${response.status}): ${errorText}`);
    }

    return response;
  }, [parsedUri]);

  const handleUriChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUri(e.target.value);
    setParsedUri(null);
    setConnectionStatus({ isConnected: false });
  };

  const checkConnection = useCallback(async () => {
    if (!uri) return;
    
    const parsed = validateAndParseUri(uri);
    if (!parsed) {
      setConnectionStatus({ 
        isConnected: false, 
        error: 'Invalid URI format. Expected: owner_id:token@host:port' 
      });
      return;
    }
    
    setIsConnecting(true);
    setParsedUri(parsed); // Set parsed URI first

    try {
      // Try health check first
      const healthResponse = await fetch(`${parsed.serverUrl}/health`, {
        headers: {
          'Authorization': `Bearer ${parsed.token}`
        }
      });
      
      if (!healthResponse.ok) {
        throw new Error('Health check failed');
      }

      // Then check readiness
      const readyResponse = await fetch(`${parsed.serverUrl}/health/ready`, {
        headers: {
          'Authorization': `Bearer ${parsed.token}`
        }
      });
      
      if (!readyResponse.ok) {
        throw new Error('Server is not ready');
      }

      setConnectionStatus({ isConnected: true });
    } catch (error) {
      console.error('Connection error:', error);
      setParsedUri(null);
      setConnectionStatus({ 
        isConnected: false, 
        error: error instanceof Error ? error.message : 'Failed to connect to DataBridge server' 
      });
    } finally {
      setIsConnecting(false);
    }
  }, [uri]);

  // Don't automatically check connection, wait for button click
  const handleConnect = () => {
    checkConnection();
  };

  const fetchDocuments = useCallback(async () => {
    if (!connectionStatus.isConnected || !parsedUri) return;
    
    try {
      const response = await makeRequest('/documents');
      const documents = await response.json();
      setSources(documents.map((doc: DocumentResponse) => ({
        id: doc.external_id,
        name: doc.filename || 'Untitled Document',
        type: doc.content_type || 'unknown',
        uploadedAt: new Date(doc.system_metadata.created_at || Date.now()),
        metadata: doc.metadata,
        filename: doc.filename
      })));
    } catch (error) {
      setConnectionStatus({ 
        isConnected: false, 
        error: error instanceof Error ? error.message : 'Failed to fetch documents' 
      });
    }
  }, [connectionStatus.isConnected, parsedUri, makeRequest]);

  // Fetch documents when connection is established
  useEffect(() => {
    if (connectionStatus.isConnected) {
      fetchDocuments();
    }
  }, [connectionStatus.isConnected, fetchDocuments]);

  const handleAddFilter = () => {
    if (newFilter.key && newFilter.value) {
      setFilters([...filters, newFilter]);
      setNewFilter({ key: '', value: '' });
    }
  };

  const handleRemoveFilter = (index: number) => {
    setFilters(filters.filter((_, i) => i !== index));
  };

  const handleAddUploadMetadata = () => {
    if (newUploadMetadata.key && newUploadMetadata.value) {
      setUploadMetadata([...uploadMetadata, newUploadMetadata]);
      setNewUploadMetadata({ key: '', value: '' });
    }
  };

  const handleRemoveUploadMetadata = (index: number) => {
    setUploadMetadata(uploadMetadata.filter((_, i) => i !== index));
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || !connectionStatus.isConnected || !parsedUri) return;
    
    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append('file', files[0]);
      
      // Convert metadata array to object
      const metadataObject = uploadMetadata.reduce((acc, meta) => {
        acc[meta.key] = meta.value;
        return acc;
      }, {} as Record<string, string>);
      
      formData.append('metadata', JSON.stringify(metadataObject));
      
      await makeRequest('/ingest/file', {
        method: 'POST',
        body: formData,
      });
      
      // Reset form
      setUploadMetadata([]);
      
      // Fetch updated document list after successful upload
      await fetchDocuments();
      setIsUploadModalOpen(false);
    } catch (error) {
      setConnectionStatus({ 
        isConnected: false, 
        error: error instanceof Error ? error.message : 'Failed to upload file' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextUpload = async () => {
    if (!textContent.trim() || !connectionStatus.isConnected || !parsedUri) return;
    
    try {
      setIsLoading(true);
      
      // Convert metadata array to object
      const metadataObject = uploadMetadata.reduce((acc, meta) => {
        acc[meta.key] = meta.value;
        return acc;
      }, {} as Record<string, string>);
      
      await makeRequest('/ingest/text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: textContent,
          metadata: metadataObject
        }),
      });
      
      // Reset form
      setTextContent('');
      setUploadMetadata([]);
      
      // Fetch updated document list
      await fetchDocuments();
      setIsUploadModalOpen(false);
    } catch (error) {
      setConnectionStatus({ 
        isConnected: false, 
        error: error instanceof Error ? error.message : 'Failed to upload text' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!message.trim() || !connectionStatus.isConnected || !parsedUri) return;
    
    try {
      setIsLoading(true);
      // Add user message immediately
      const userMessage: Message = {
        id: Date.now().toString(),
        content: message,
        role: 'user',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, userMessage]);
      setMessage(''); // Clear input immediately
      
      // Convert filters array to object
      const filterObject = filters.reduce((acc, filter) => {
        acc[filter.key] = filter.value;
        return acc;
      }, {} as Record<string, string>);

      const response = await makeRequest('/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          filters: Object.keys(filterObject).length > 0 ? filterObject : undefined,
          max_tokens: 1000,
          temperature: 0.7,
        }),
      });
      
      const data = await response.json();
      console.log('Response data:', data);
      
      // Add assistant message
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.completion || data.text || data.answer || 'No response',
        role: 'assistant',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      setConnectionStatus({ 
        isConnected: false, 
        error: error instanceof Error ? error.message : 'Failed to send message' 
      });
      setMessage(message); // Restore message on error
    } finally {
      setIsLoading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const files = e.dataTransfer.files;
    handleFileUpload(files);
  };

  return (
    <div className="flex flex-col h-screen bg-background dark text-foreground">
      {/* Top Navigation Bar */}
      <div className="flex items-center justify-between px-4 h-14 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex items-center gap-4">
          {isEditingTitle ? (
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onBlur={() => setIsEditingTitle(false)}
              onKeyDown={(e) => e.key === 'Enter' && setIsEditingTitle(false)}
              className="h-8 w-[300px]"
              autoFocus
            />
          ) : (
            <h1
              className="text-lg font-semibold cursor-pointer hover:text-primary/80"
              onClick={() => setIsEditingTitle(true)}
            >
              {title}
            </h1>
          )}
        </div>
        
        {/* Connection Controls */}
        <div className="flex items-center gap-2">
          <Input
            value={uri}
            onChange={handleUriChange}
            placeholder="DataBridge URI (e.g. http://test_user:token@localhost:8000)"
            className="h-8 w-[400px] font-mono"
          />
          <Button 
            onClick={handleConnect}
            disabled={!uri || isConnecting}
            size="sm"
            className="h-8"
          >
            {isConnecting ? 'Connecting...' : 'Connect'}
          </Button>
          {connectionStatus.isConnected ? (
            <div className="flex items-center text-sm text-green-500">
              <CheckCircle2 className="h-4 w-4 mr-1" />
              Connected to {parsedUri?.serverUrl}
            </div>
          ) : connectionStatus.error ? (
            <div className="flex items-center text-sm text-destructive">
              {connectionStatus.error}
            </div>
          ) : null}
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel - Sources */}
        <div 
          className={`flex flex-col border-r transition-all duration-300 relative ${
            isPanelCollapsed ? 'w-14' : ''
          }`}
          style={{ width: isPanelCollapsed ? '3.5rem' : `${panelWidth}px` }}
        >
          <div className="flex items-center justify-between p-4 border-b">
            {!isPanelCollapsed && <h2 className="text-lg font-semibold text-foreground">Sources</h2>}
            <div className="flex items-center gap-2">
              {!isPanelCollapsed && (
                <Dialog open={isUploadModalOpen} onOpenChange={setIsUploadModalOpen}>
                  <DialogTrigger asChild>
                    <Button size="sm" className="flex items-center gap-2">
                      <Plus className="h-4 w-4" /> Add Source
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-3xl">
                    <DialogHeader>
                      <DialogTitle className="text-foreground">Add Source</DialogTitle>
                    </DialogHeader>
                    <Tabs defaultValue="file">
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="file">Upload File</TabsTrigger>
                        <TabsTrigger value="text">Add Text</TabsTrigger>
                      </TabsList>
                      <TabsContent value="file">
                        <div
                          className="grid gap-4 py-4"
                          onDragOver={handleDragOver}
                          onDrop={handleDrop}
                        >
                          <div
                            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary/50 transition-colors ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
                            onClick={() => document.getElementById('file-upload')?.click()}
                          >
                            {isLoading ? (
                              <>
                                <div className="h-8 w-8 mx-auto mb-2 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                                <p className="text-sm text-muted-foreground mb-1">
                                  Uploading and processing your file...
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  This may take a few moments
                                </p>
                              </>
                            ) : (
                              <>
                                <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                                <p className="text-sm text-muted-foreground mb-1">
                                  Drag and drop your files here or click to browse
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  Supported formats: PDF, TXT, MD, MP3, MP4, AVI, MOV, MKV, FLV, WEBM, M4V, MPEG, MPG
                                </p>
                              </>
                            )}
                            <input
                              id="file-upload"
                              type="file"
                              className="hidden"
                              accept=".pdf,.txt,.md,.mp3,.mp4,.avi,.mov,.mkv,.wmv,.flv,.webm,.m4v,.mpeg,.mpg"
                              onChange={(e) => handleFileUpload(e.target.files)}
                              disabled={isLoading}
                            />
                          </div>
                        </div>
                      </TabsContent>
                      <TabsContent value="text">
                        <div className="grid gap-4 py-4">
                          <Textarea
                            placeholder="Enter your text content here..."
                            value={textContent}
                            onChange={(e) => setTextContent(e.target.value)}
                            className="min-h-[200px]"
                            disabled={isLoading}
                          />
                          <Button 
                            onClick={handleTextUpload} 
                            disabled={!textContent.trim() || isLoading}
                            className="w-full"
                          >
                            {isLoading ? (
                              <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                            ) : (
                              <>
                                <Upload className="h-4 w-4 mr-2" />
                                Add Text
                              </>
                            )}
                          </Button>
                        </div>
                      </TabsContent>
                      <div className="border-t mt-4 pt-4">
                        <h4 className="text-sm font-medium mb-2">Metadata (Optional)</h4>
                        <div className="space-y-4">
                          <div className="flex gap-2">
                            <Input
                              placeholder="Key"
                              value={newUploadMetadata.key}
                              onChange={(e) => setNewUploadMetadata({ ...newUploadMetadata, key: e.target.value })}
                              disabled={isLoading}
                            />
                            <Input
                              placeholder="Value"
                              value={newUploadMetadata.value}
                              onChange={(e) => setNewUploadMetadata({ ...newUploadMetadata, value: e.target.value })}
                              disabled={isLoading}
                            />
                            <Button 
                              onClick={handleAddUploadMetadata} 
                              disabled={!newUploadMetadata.key || !newUploadMetadata.value || isLoading}
                            >
                              Add
                            </Button>
                          </div>
                          <div className="space-y-2">
                            {uploadMetadata.map((meta, index) => (
                              <div key={index} className="flex items-center gap-2 bg-muted p-2 rounded-md">
                                <span className="flex-1">
                                  {meta.key}: {meta.value}
                                </span>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleRemoveUploadMetadata(index)}
                                  className="h-8 w-8 hover:bg-destructive/90 hover:text-destructive-foreground"
                                  disabled={isLoading}
                                >
                                  <X className="h-4 w-4" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </Tabs>
                  </DialogContent>
                </Dialog>
              )}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsPanelCollapsed(!isPanelCollapsed)}
                className="h-8 w-8"
              >
                {isPanelCollapsed ? (
                  <ChevronRight className="h-4 w-4" />
                ) : (
                  <ChevronLeft className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
          <ScrollArea className="flex-1">
            <div className="p-4 space-y-2">
              {!isPanelCollapsed && sources.map((source) => (
                <Card key={source.id} className="p-3 bg-card hover:bg-card/80">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-card-foreground">{source.filename || 'Untitled Document'}</p>
                      <p className="text-xs text-muted-foreground">
                        {source.type} • {source.uploadedAt.toLocaleDateString()}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 hover:bg-destructive/90 hover:text-destructive-foreground"
                      onClick={() => {
                        setSources(sources.filter(s => s.id !== source.id));
                      }}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          </ScrollArea>
          
          {/* Resize Handle */}
          {!isPanelCollapsed && (
            <div
              className="absolute right-0 top-0 bottom-0 w-1 cursor-ew-resize hover:bg-primary/10 active:bg-primary/20"
              onMouseDown={handleMouseDown}
            />
          )}
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-hidden">
            {sources.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <Upload className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium text-foreground mb-2">Add a source to get started</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Upload your documents to start chatting with your knowledge base
                </p>
                <Dialog open={isUploadModalOpen} onOpenChange={setIsUploadModalOpen}>
                  <DialogTrigger asChild>
                    <Button>
                      <Upload className="h-4 w-4 mr-2" /> Upload Source
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-3xl">
                    <DialogHeader>
                      <DialogTitle className="text-foreground">Add Source</DialogTitle>
                    </DialogHeader>
                    <Tabs defaultValue="file">
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="file">Upload File</TabsTrigger>
                        <TabsTrigger value="text">Add Text</TabsTrigger>
                      </TabsList>
                      <TabsContent value="file">
                        <div
                          className="grid gap-4 py-4"
                          onDragOver={handleDragOver}
                          onDrop={handleDrop}
                        >
                          <div
                            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary/50 transition-colors ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
                            onClick={() => document.getElementById('file-upload')?.click()}
                          >
                            {isLoading ? (
                              <>
                                <div className="h-8 w-8 mx-auto mb-2 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                                <p className="text-sm text-muted-foreground mb-1">
                                  Uploading and processing your file...
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  This may take a few moments
                                </p>
                              </>
                            ) : (
                              <>
                                <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                                <p className="text-sm text-muted-foreground mb-1">
                                  Drag and drop your files here or click to browse
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  Supported formats: PDF, TXT, MD, MP3, MP4, AVI, MOV, MKV, FLV, WEBM, M4V, MPEG, MPG
                                </p>
                              </>
                            )}
                            <input
                              id="file-upload"
                              type="file"
                              className="hidden"
                              accept=".pdf,.txt,.md,.mp3,.mp4,.avi,.mov,.mkv,.wmv,.flv,.webm,.m4v,.mpeg,.mpg"
                              onChange={(e) => handleFileUpload(e.target.files)}
                              disabled={isLoading}
                            />
                          </div>
                        </div>
                      </TabsContent>
                      <TabsContent value="text">
                        <div className="grid gap-4 py-4">
                          <Textarea
                            placeholder="Enter your text content here..."
                            value={textContent}
                            onChange={(e) => setTextContent(e.target.value)}
                            className="min-h-[200px]"
                            disabled={isLoading}
                          />
                          <Button 
                            onClick={handleTextUpload} 
                            disabled={!textContent.trim() || isLoading}
                            className="w-full"
                          >
                            {isLoading ? (
                              <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                            ) : (
                              <>
                                <Upload className="h-4 w-4 mr-2" />
                                Add Text
                              </>
                            )}
                          </Button>
                        </div>
                      </TabsContent>
                      <div className="border-t mt-4 pt-4">
                        <h4 className="text-sm font-medium mb-2">Metadata (Optional)</h4>
                        <div className="space-y-4">
                          <div className="flex gap-2">
                            <Input
                              placeholder="Key"
                              value={newUploadMetadata.key}
                              onChange={(e) => setNewUploadMetadata({ ...newUploadMetadata, key: e.target.value })}
                              disabled={isLoading}
                            />
                            <Input
                              placeholder="Value"
                              value={newUploadMetadata.value}
                              onChange={(e) => setNewUploadMetadata({ ...newUploadMetadata, value: e.target.value })}
                              disabled={isLoading}
                            />
                            <Button 
                              onClick={handleAddUploadMetadata} 
                              disabled={!newUploadMetadata.key || !newUploadMetadata.value || isLoading}
                            >
                              Add
                            </Button>
                          </div>
                          <div className="space-y-2">
                            {uploadMetadata.map((meta, index) => (
                              <div key={index} className="flex items-center gap-2 bg-muted p-2 rounded-md">
                                <span className="flex-1">
                                  {meta.key}: {meta.value}
                                </span>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleRemoveUploadMetadata(index)}
                                  className="h-8 w-8 hover:bg-destructive/90 hover:text-destructive-foreground"
                                  disabled={isLoading}
                                >
                                  <X className="h-4 w-4" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </Tabs>
                  </DialogContent>
                </Dialog>
              </div>
            ) : (
              <ScrollArea className="h-full">
                <div className="flex flex-col space-y-4 p-4">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${
                        msg.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg px-4 py-2 ${
                          msg.role === 'user'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted'
                        }`}
                      >
                        <p className="text-sm whitespace-pre-wrap break-words">{msg.content}</p>
                        <p className="text-xs opacity-70 mt-1">
                          {msg.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </div>
          
          {/* Chat Input - Fixed at bottom */}
          <div className="flex-none p-4 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="flex gap-2 max-w-5xl mx-auto">
              <Dialog open={isFilterModalOpen} onOpenChange={setIsFilterModalOpen}>
                <DialogTrigger asChild>
                  <Button 
                    variant="outline" 
                    size="icon"
                    disabled={!connectionStatus.isConnected || sources.length === 0}
                    className="shrink-0"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-4 w-4"
                    >
                      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
                    </svg>
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="text-foreground">Metadata Filters</DialogTitle>
                  </DialogHeader>
                  <div className="grid gap-4 py-4">
                    <div className="flex gap-2">
                      <Input
                        placeholder="Key"
                        value={newFilter.key}
                        onChange={(e) => setNewFilter({ ...newFilter, key: e.target.value })}
                      />
                      <Input
                        placeholder="Value"
                        value={newFilter.value}
                        onChange={(e) => setNewFilter({ ...newFilter, value: e.target.value })}
                      />
                      <Button onClick={handleAddFilter} disabled={!newFilter.key || !newFilter.value}>
                        Add
                      </Button>
                    </div>
                    <div className="space-y-2">
                      {filters.map((filter, index) => (
                        <div key={index} className="flex items-center gap-2 bg-muted p-2 rounded-md">
                          <span className="flex-1">
                            {filter.key}: {filter.value}
                          </span>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleRemoveFilter(index)}
                            className="h-8 w-8 hover:bg-destructive/90 hover:text-destructive-foreground"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
              <Input
                placeholder={
                  !connectionStatus.isConnected
                    ? "Connect to start chatting..."
                    : sources.length === 0
                    ? "Upload a source to start chatting..."
                    : "Type your message..."
                }
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                disabled={!connectionStatus.isConnected || sources.length === 0 || isLoading}
                className="text-foreground"
              />
              <Button 
                disabled={!connectionStatus.isConnected || sources.length === 0 || !message.trim() || isLoading}
                onClick={handleSendMessage}
              >
                {isLoading ? (
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
            {filters.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2 max-w-5xl mx-auto">
                {filters.map((filter, index) => (
                  <div key={index} className="flex items-center gap-1 bg-muted px-2 py-1 rounded text-xs">
                    <span>
                      {filter.key}: {filter.value}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleRemoveFilter(index)}
                      className="h-4 w-4 hover:bg-destructive/90 hover:text-destructive-foreground p-0"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 
