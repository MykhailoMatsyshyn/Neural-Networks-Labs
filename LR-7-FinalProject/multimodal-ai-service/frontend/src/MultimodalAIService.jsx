import React, { useState, useEffect, useRef } from "react";
import {
  Send,
  Plus,
  Settings,
  Image,
  FileText,
  Calendar,
  Mail,
  DollarSign,
  Trash2,
  Download,
  Upload,
  Sparkles,
  MessageSquare,
  Bot,
  User,
  Loader2,
  X,
} from "lucide-react";

const MultimodalAIService = () => {
  // ==================== STATE ====================
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [mode, setMode] = useState("chat"); // chat, image-gen, image-analyze, photo, rag
  const [photoSubMode, setPhotoSubMode] = useState("generate"); // generate, analyze
  const [gallery, setGallery] = useState([]);
  const [showGallery, setShowGallery] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null); // Для модалки перегляду зображення
  const [ragFiles, setRagFiles] = useState([]); // Список завантажених файлів для RAG
  const [ragProcessing, setRagProcessing] = useState(false); // Стан обробки RAG
  const [ragVisualization, setRagVisualization] = useState(null); // Дані для візуалізації RAG процесу
  const [settings, setSettings] = useState({
    model: "gpt-4o-mini",
    temperature: 0.7,
    enableRAG: true,
    enableAgent: true,
    enableStreaming: true, // Новий параметр для streaming
    imageModel: "dall-e-3",
    imageSettings: {
      model: "dall-e-3",
      size: "1024x1024",
      quality: "standard",
      style: "vivid",
    },
    detailedAnalysis: true,
    enabledTools: {
      get_item_price: true,
      calculate_shipping: true,
      book_meeting: true,
      send_email: true,
    },
  });
  const [streamingContent, setStreamingContent] = useState(""); // Для streaming
  const [showSettings, setShowSettings] = useState(false);
  const chatEndRef = useRef(null);

  // ==================== STORAGE ====================
  useEffect(() => {
    const saved = localStorage.getItem("ai_threads");
    if (saved) {
      const parsed = JSON.parse(saved);
      setThreads(parsed);
      if (parsed.length > 0) setActiveThreadId(parsed[0].id);
    } else {
      createNewThread();
    }
  }, []);

  useEffect(() => {
    if (threads.length > 0) {
      localStorage.setItem("ai_threads", JSON.stringify(threads));
    }
  }, [threads]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [threads, activeThreadId]);

  // ==================== THREAD MANAGEMENT ====================
  const createNewThread = () => {
    const newThread = {
      id: Date.now(),
      name: `Thread ${threads.length + 1}`,
      messages: [],
      createdAt: new Date().toISOString(),
    };
    setThreads((prev) => [newThread, ...prev]);
    setActiveThreadId(newThread.id);
  };

  const deleteThread = (id) => {
    setThreads((prev) => prev.filter((t) => t.id !== id));
    if (activeThreadId === id && threads.length > 1) {
      const remaining = threads.filter((t) => t.id !== id);
      setActiveThreadId(remaining[0]?.id);
    }
  };

  const renameThread = (id, newName) => {
    setThreads((prev) =>
      prev.map((t) => (t.id === id ? { ...t, name: newName } : t))
    );
  };

  const activeThread = threads.find((t) => t.id === activeThreadId);

  // ==================== API CONFIG ====================
  const API_BASE_URL = "http://localhost:8000";

  // ==================== HELPER FUNCTIONS ====================
  const convertImageToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(",")[1]; // Remove data:image/...;base64, prefix
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  // Завантажити галерею
  const loadGallery = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/gallery`);
      if (response.ok) {
        const data = await response.json();
        setGallery(data.gallery || []);
      }
    } catch (error) {
      console.error("Error loading gallery:", error);
    }
  };

  // Завантажити галерею при завантаженні компонента
  useEffect(() => {
    if (mode === "photo") {
      loadGallery();
    }
  }, [mode]);

  // Завантажити список файлів RAG
  const loadRagFiles = async () => {
    // TODO: Додати endpoint для отримання списку файлів
    // Поки що використовуємо локальний стан
  };

  // Завантажити файли в RAG
  const uploadRagFiles = async (files) => {
    if (!files || files.length === 0) return;

    setRagProcessing(true);
    try {
      const formData = new FormData();
      Array.from(files).forEach((file) => {
        formData.append("files", file);
      });
      formData.append("thread_id", activeThreadId?.toString() || "");

      const response = await fetch(`${API_BASE_URL}/upload_documents`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "error") {
        throw new Error(data.message || "Помилка завантаження файлів");
      }

      // Оновити список файлів
      const newFiles = Array.from(files)
        .filter((file) => {
          // Фільтрувати тільки успішно завантажені файли
          return data.files?.includes(file.name);
        })
        .map((file) => ({
          id: Date.now() + Math.random(),
          name: file.name,
          size: file.size,
          type: file.type,
          uploadedAt: new Date().toISOString(),
          method: data.method || "unknown",
        }));

      setRagFiles((prev) => [...prev, ...newFiles]);

      // Показати попередження якщо є
      if (data.warnings && data.warnings.length > 0) {
        console.warn("Warnings during upload:", data.warnings);
        alert(`Попередження: ${data.warnings.join(", ")}`);
      }

      return data;
    } catch (error) {
      console.error("Error uploading files:", error);
      const errorMessage =
        error.message || "Невідома помилка завантаження файлів";
      alert(`Помилка завантаження: ${errorMessage}`);
      throw error;
    } finally {
      setRagProcessing(false);
    }
  };

  // Format text with markdown-style links [text](url)
  const formatMessageContent = (text) => {
    if (!text) return "";

    // Regular expression to match [text](url) pattern
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = linkRegex.exec(text)) !== null) {
      // Add text before the link
      if (match.index > lastIndex) {
        parts.push({
          type: "text",
          content: text.substring(lastIndex, match.index),
        });
      }
      // Add the link
      parts.push({
        type: "link",
        text: match[1],
        url: match[2],
      });
      lastIndex = match.index + match[0].length;
    }

    // Add remaining text after the last link
    if (lastIndex < text.length) {
      parts.push({
        type: "text",
        content: text.substring(lastIndex),
      });
    }

    // If no links found, return original text as React element
    if (parts.length === 0) {
      return <>{text}</>;
    }

    // Render parts as React elements
    return parts.map((part, idx) => {
      if (part.type === "link") {
        return (
          <a
            key={idx}
            href={part.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-purple-300 hover:text-purple-200 underline"
          >
            {part.text}
          </a>
        );
      }
      return <span key={idx}>{part.content}</span>;
    });
  };

  // ==================== AI LOGIC ====================
  const handleSend = async () => {
    if (!input.trim() && !imageFile) return;

    // Визначити режим для Photo вкладки та RAG
    let actualMode = mode;
    if (mode === "photo") {
      actualMode = photoSubMode === "generate" ? "image-gen" : "image-analyze";
    } else if (mode === "rag") {
      actualMode = "rag"; // Явно встановити режим RAG
    }

    const userMsg = {
      role: "user",
      content: input,
      image: imageFile ? URL.createObjectURL(imageFile) : null,
      timestamp: new Date().toISOString(),
      messageMode: actualMode, // Зберігаємо режим повідомлення
    };

    setThreads((prev) =>
      prev.map((t) =>
        t.id === activeThreadId
          ? { ...t, messages: [...t.messages, userMsg] }
          : t
      )
    );

    const currentInput = input;
    const currentImageFile = imageFile;

    // Визначити чи буде streaming
    const useStreaming =
      settings.enableStreaming &&
      (mode === "chat" ||
        mode === "rag" ||
        (mode === "photo" && photoSubMode === "analyze")) &&
      settings.enableAgent &&
      !currentImageFile;

    setInput("");
    setImageFile(null);
    setStreamingContent(""); // Очистити streaming контент

    // Встановити loading тільки якщо не streaming
    setLoading(!useStreaming);

    try {
      // Конвертувати зображення в base64, якщо є
      let imageBase64 = null;
      if (currentImageFile) {
        imageBase64 = await convertImageToBase64(currentImageFile);
      }

      // Підготувати історію повідомлень (останні N повідомлень для контексту)
      const messageHistory =
        activeThread?.messages
          ?.slice(0, -1)
          .slice(-10)
          .map((msg) => ({
            role: msg.role,
            content: msg.content || "",
          }))
          .filter((msg) => msg.content.trim().length > 0) || [];

      // actualMode вже визначено вище

      // Підготувати запит до backend
      const requestBody = {
        thread_id: activeThreadId.toString(),
        message:
          currentInput ||
          (actualMode === "image-analyze" ? "Analyze this image" : ""),
        mode: actualMode,
        image_base64: imageBase64,
        settings: settings,
        history: messageHistory,
      };

      // useStreaming вже визначено вище

      if (useStreaming) {
        // ===== STREAMING MODE =====
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Створити placeholder повідомлення для streaming
        const streamingMsgId = Date.now();
        const streamingMsg = {
          id: streamingMsgId,
          role: "assistant",
          content: "",
          tools: [],
          timestamp: new Date().toISOString(),
          isStreaming: true,
          messageMode: actualMode, // Зберігаємо режим повідомлення
        };

        setThreads((prev) =>
          prev.map((t) =>
            t.id === activeThreadId
              ? { ...t, messages: [...t.messages, streamingMsg] }
              : t
          )
        );

        // Читати stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = "";
        let buffer = ""; // Буфер для неповних рядків

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Декодувати chunk та додати до буфера
            buffer += decoder.decode(value, { stream: true });

            // Обробити повні рядки
            const lines = buffer.split("\n");
            // Залишити останній неповний рядок в буфері
            buffer = lines.pop() || "";

            for (const line of lines) {
              const trimmedLine = line.trim();
              if (!trimmedLine || !trimmedLine.startsWith("data: ")) continue;

              try {
                const jsonStr = trimmedLine.slice(6); // Видалити "data: "
                if (!jsonStr) continue;

                const data = JSON.parse(jsonStr);

                if (data.error) {
                  throw new Error(data.error);
                }

                if (data.content) {
                  fullContent += data.content;
                  // Оновити повідомлення в реальному часі
                  setThreads((prev) =>
                    prev.map((t) =>
                      t.id === activeThreadId
                        ? {
                            ...t,
                            messages: t.messages.map((msg) =>
                              msg.id === streamingMsgId
                                ? { ...msg, content: fullContent }
                                : msg
                            ),
                          }
                        : t
                    )
                  );
                }

                if (data.done) {
                  // Завершити streaming
                  const finalContent = data.full_content || fullContent;

                  // Якщо є tool calls, потрібно зробити додатковий запит
                  if (data.has_tools) {
                    // Видалити streaming повідомлення та зробити звичайний запит
                    setThreads((prev) =>
                      prev.map((t) =>
                        t.id === activeThreadId
                          ? {
                              ...t,
                              messages: t.messages.filter(
                                (msg) => msg.id !== streamingMsgId
                              ),
                            }
                          : t
                      )
                    );

                    // Зробити звичайний non-streaming запит
                    const regularResponse = await fetch(
                      `${API_BASE_URL}/chat`,
                      {
                        method: "POST",
                        headers: {
                          "Content-Type": "application/json",
                        },
                        body: JSON.stringify(requestBody),
                      }
                    );

                    if (regularResponse.ok) {
                      const regularData = await regularResponse.json();
                      const aiMsg = {
                        role: "assistant",
                        content: regularData.content,
                        tools: regularData.tools || [],
                        timestamp: new Date().toISOString(),
                        messageMode: actualMode, // Зберігаємо режим повідомлення
                      };

                      setThreads((prev) =>
                        prev.map((t) =>
                          t.id === activeThreadId
                            ? { ...t, messages: [...t.messages, aiMsg] }
                            : t
                        )
                      );
                    }
                  } else {
                    // Звичайне завершення streaming
                    setThreads((prev) =>
                      prev.map((t) =>
                        t.id === activeThreadId
                          ? {
                              ...t,
                              messages: t.messages.map((msg) =>
                                msg.id === streamingMsgId
                                  ? {
                                      ...msg,
                                      content: finalContent,
                                      isStreaming: false,
                                      messageMode: actualMode, // Зберігаємо режим повідомлення
                                    }
                                  : msg
                              ),
                            }
                          : t
                      )
                    );
                  }
                  break;
                }
              } catch (e) {
                // Ігнорувати помилки парсингу окремих рядків
                if (e instanceof SyntaxError) {
                  console.warn("Failed to parse SSE line:", trimmedLine);
                } else {
                  console.error("Error parsing stream data:", e);
                }
              }
            }
          }

          // Обробити залишок буфера
          if (buffer.trim()) {
            const trimmedLine = buffer.trim();
            if (trimmedLine.startsWith("data: ")) {
              try {
                const jsonStr = trimmedLine.slice(6);
                const data = JSON.parse(jsonStr);
                if (data.content) {
                  fullContent += data.content;
                }
                if (data.done) {
                  const finalContent = data.full_content || fullContent;
                  setThreads((prev) =>
                    prev.map((t) =>
                      t.id === activeThreadId
                        ? {
                            ...t,
                            messages: t.messages.map((msg) =>
                              msg.id === streamingMsgId
                                ? {
                                    ...msg,
                                    content: finalContent,
                                    isStreaming: false,
                                  }
                                : msg
                            ),
                          }
                        : t
                    )
                  );
                }
              } catch (e) {
                console.error("Error parsing final buffer:", e);
              }
            }
          }
        } catch (error) {
          console.error("Streaming error:", error);
          // Завершити streaming з помилкою
          setThreads((prev) =>
            prev.map((t) =>
              t.id === activeThreadId
                ? {
                    ...t,
                    messages: t.messages.map((msg) =>
                      msg.id === streamingMsgId
                        ? {
                            ...msg,
                            content:
                              fullContent ||
                              `Помилка streaming: ${error.message}`,
                            isStreaming: false,
                          }
                        : msg
                    ),
                  }
                : t
            )
          );
        }
      } else {
        // ===== REGULAR MODE (non-streaming) =====
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        const aiMsg = {
          role: "assistant",
          content: data.content,
          tools: data.tools || [],
          image_url: data.image_url,
          timestamp: new Date().toISOString(),
          messageMode: actualMode, // Зберігаємо режим повідомлення
        };

        setThreads((prev) =>
          prev.map((t) =>
            t.id === activeThreadId
              ? { ...t, messages: [...t.messages, aiMsg] }
              : t
          )
        );

        // Оновити галерею якщо згенеровано зображення
        if (
          data.image_url &&
          (actualMode === "image-gen" || mode === "photo")
        ) {
          loadGallery();
        }
      }
    } catch (error) {
      console.error("Error calling API:", error);
      const errorMsg = {
        role: "assistant",
        content: `Помилка: ${error.message}. Переконайтеся, що backend запущено на ${API_BASE_URL}`,
        tools: [],
        timestamp: new Date().toISOString(),
      };

      setThreads((prev) =>
        prev.map((t) =>
          t.id === activeThreadId
            ? { ...t, messages: [...t.messages, errorMsg] }
            : t
        )
      );
    } finally {
      setLoading(false);
      setStreamingContent("");
    }
  };

  // ==================== RENDER ====================
  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* SIDEBAR */}
      <div className="w-72 bg-black/30 backdrop-blur-xl border-r border-white/10 flex flex-col">
        <div className="p-4 border-b border-white/10">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Bot className="w-6 h-6 text-purple-400" />
            AI Platform
          </h1>
        </div>

        <div className="p-4">
          <button
            onClick={createNewThread}
            className="w-full bg-purple-600 hover:bg-purple-700 rounded-lg p-3 flex items-center justify-center gap-2 transition"
          >
            <Plus className="w-5 h-5" />
            New Thread
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {threads.map((thread) => (
            <div
              key={thread.id}
              onClick={() => setActiveThreadId(thread.id)}
              className={`p-3 rounded-lg cursor-pointer transition group ${
                activeThreadId === thread.id
                  ? "bg-purple-600/50 border border-purple-400"
                  : "bg-white/5 hover:bg-white/10"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <MessageSquare className="w-4 h-4 flex-shrink-0" />
                  <span className="text-sm truncate">{thread.name}</span>
                </div>
                {threads.length > 1 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteThread(thread.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {thread.messages.length} messages
              </div>
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-white/10 space-y-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="w-full bg-white/5 hover:bg-white/10 rounded-lg p-3 flex items-center gap-2 transition"
          >
            <Settings className="w-5 h-5" />
            Settings
          </button>
        </div>
      </div>

      {/* MAIN CHAT */}
      <div className="flex-1 flex flex-col">
        {/* HEADER */}
        <div className="h-16 bg-black/30 backdrop-blur-xl border-b border-white/10 flex items-center justify-between px-6">
          <div>
            <h2 className="font-semibold">
              {activeThread?.name || "No thread"}
            </h2>
            <p className="text-xs text-gray-400">
              {activeThread?.messages.length || 0} messages
            </p>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setMode("chat")}
              className={`px-4 py-2 rounded-lg transition ${
                mode === "chat"
                  ? "bg-purple-600"
                  : "bg-white/5 hover:bg-white/10"
              }`}
            >
              💬 Chat
            </button>
            <button
              onClick={() => {
                setMode("photo");
                setPhotoSubMode("generate");
              }}
              className={`px-4 py-2 rounded-lg transition ${
                mode === "photo"
                  ? "bg-purple-600"
                  : "bg-white/5 hover:bg-white/10"
              }`}
            >
              📸 Photo
            </button>
            <button
              onClick={() => setMode("rag")}
              className={`px-4 py-2 rounded-lg transition ${
                mode === "rag"
                  ? "bg-purple-600"
                  : "bg-white/5 hover:bg-white/10"
              }`}
            >
              📚 RAG
            </button>
          </div>
        </div>

        {/* RAG MODE UI */}
        {mode === "rag" && (
          <div className="flex-1 flex min-h-0">
            {/* LEFT SIDEBAR - Files & Settings */}
            <div className="w-80 bg-black/20 border-r border-white/10 flex flex-col min-h-0">
              <div className="flex-shrink-0 p-4 border-b border-white/10">
                <h3 className="text-lg font-semibold mb-3">
                  📚 Knowledge Base
                </h3>

                {/* Upload Files */}
                <label
                  className={`cursor-pointer block w-full rounded-lg p-3 text-center transition mb-4 ${
                    ragProcessing
                      ? "bg-gray-600 cursor-wait"
                      : "bg-purple-600 hover:bg-purple-700"
                  }`}
                >
                  {ragProcessing ? (
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span className="text-sm font-medium">Uploading...</span>
                    </div>
                  ) : (
                    <>
                      <Upload className="w-5 h-5 inline-block mr-2" />
                      <span className="text-sm font-medium">
                        Upload Documents
                      </span>
                    </>
                  )}
                  <input
                    type="file"
                    className="hidden"
                    multiple
                    accept=".txt,.pdf,.md,.doc,.docx"
                    disabled={ragProcessing}
                    onChange={async (e) => {
                      const files = e.target.files;
                      if (files && files.length > 0) {
                        try {
                          await uploadRagFiles(files);
                        } catch (error) {
                          alert(`Помилка завантаження: ${error.message}`);
                        }
                      }
                    }}
                  />
                </label>

                {/* RAG Status */}
                <div className="bg-white/5 rounded-lg p-3 mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">RAG Status</span>
                    <span
                      className={`text-xs px-2 py-1 rounded ${
                        settings.enableRAG
                          ? "bg-green-500/20 text-green-400"
                          : "bg-gray-500/20 text-gray-400"
                      }`}
                    >
                      {settings.enableRAG ? "Active" : "Disabled"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>Files:</span>
                    <span className="text-white">{ragFiles.length}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-400 mt-1">
                    <span>Method:</span>
                    <span className="text-white">
                      {ragFiles.length > 0 &&
                      ragFiles[0]?.method === "openai_file_search"
                        ? "OpenAI File Search"
                        : ragFiles.length > 0
                        ? "ChromaDB"
                        : "None"}
                    </span>
                  </div>
                </div>

                {/* Settings Toggle */}
                <label className="flex items-center gap-2 bg-white/5 rounded-lg p-3 cursor-pointer hover:bg-white/10 transition">
                  <input
                    type="checkbox"
                    checked={settings.enableRAG}
                    onChange={(e) =>
                      setSettings({ ...settings, enableRAG: e.target.checked })
                    }
                    className="rounded"
                  />
                  <span className="text-sm">Enable RAG</span>
                </label>
              </div>

              {/* Files List */}
              <div className="flex-1 overflow-y-auto p-4 min-h-0">
                <h4 className="text-sm font-semibold mb-3 text-gray-400">
                  Uploaded Files ({ragFiles.length})
                </h4>
                {ragFiles.length === 0 ? (
                  <div className="text-center text-gray-500 text-sm mt-8">
                    No files uploaded yet.
                    <br />
                    Upload documents to build your knowledge base.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {ragFiles.map((file) => (
                      <div
                        key={file.id}
                        className="bg-white/5 rounded-lg p-3 hover:bg-white/10 transition"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <FileText className="w-4 h-4 text-purple-400 flex-shrink-0" />
                              <span className="text-sm font-medium truncate">
                                {file.name}
                              </span>
                            </div>
                            <div className="text-xs text-gray-400">
                              {(file.size / 1024).toFixed(1)} KB
                            </div>
                            <div className="text-xs text-purple-400 mt-1">
                              {file.method === "openai_file_search"
                                ? "🔷 OpenAI File Search"
                                : "🔶 ChromaDB"}
                            </div>
                          </div>
                          <button
                            onClick={() => {
                              setRagFiles((prev) =>
                                prev.filter((f) => f.id !== file.id)
                              );
                            }}
                            className="text-red-400 hover:text-red-300 ml-2"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* RIGHT SIDE - Chat */}
            <div className="flex-1 flex flex-col min-h-0">
              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">
                {activeThread?.messages
                  .filter((msg) => {
                    // Показувати тільки повідомлення з RAG режиму
                    return msg.messageMode === "rag";
                  })
                  .map((msg, idx) => (
                    <div
                      key={idx}
                      className={`flex gap-3 ${
                        msg.role === "user" ? "justify-end" : "justify-start"
                      }`}
                    >
                      {msg.role === "assistant" && (
                        <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                          <Bot className="w-5 h-5" />
                        </div>
                      )}

                      <div
                        className={`max-w-2xl rounded-2xl p-4 ${
                          msg.role === "user"
                            ? "bg-purple-600"
                            : "bg-white/10 backdrop-blur-xl"
                        }`}
                      >
                        <div className="whitespace-pre-wrap">
                          {formatMessageContent(msg.content)}
                          {msg.isStreaming && (
                            <span className="inline-block w-0.5 h-5 bg-purple-400 ml-1.5 align-middle streaming-cursor" />
                          )}
                        </div>

                        {/* RAG Visualization */}
                        {msg.tools?.some((t) => t.type === "rag") && (
                          <div className="mt-4 p-4 bg-purple-500/10 rounded-lg border border-purple-500/30">
                            <div className="flex items-center gap-2 mb-3">
                              <FileText className="w-4 h-4 text-purple-400" />
                              <span className="text-sm font-semibold text-purple-300">
                                RAG Pipeline Active
                              </span>
                            </div>

                            {/* RAG Process Steps */}
                            <div className="space-y-3">
                              {/* Step 1: Query Embedding */}
                              <div className="flex items-start gap-3">
                                <div className="w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                                  <span className="text-xs text-purple-300">
                                    1
                                  </span>
                                </div>
                                <div className="flex-1">
                                  <div className="text-xs font-semibold text-purple-300 mb-1">
                                    Query Embedding
                                  </div>
                                  <div className="text-xs text-gray-400">
                                    Converting user query to vector using{" "}
                                    <span className="text-purple-400">
                                      text-embedding-3-small
                                    </span>
                                  </div>
                                </div>
                              </div>

                              {/* Step 2: Vector Search */}
                              <div className="flex items-start gap-3">
                                <div className="w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                                  <span className="text-xs text-purple-300">
                                    2
                                  </span>
                                </div>
                                <div className="flex-1">
                                  <div className="text-xs font-semibold text-purple-300 mb-1">
                                    Vector Search
                                  </div>
                                  <div className="text-xs text-gray-400">
                                    Searching knowledge base for relevant
                                    documents
                                  </div>
                                  {msg.tools?.find((t) => t.type === "rag")
                                    ?.docs && (
                                    <div className="mt-2 flex flex-wrap gap-2">
                                      {msg.tools
                                        .find((t) => t.type === "rag")
                                        .docs.map((doc, i) => (
                                          <span
                                            key={i}
                                            className="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded"
                                          >
                                            {doc}
                                          </span>
                                        ))}
                                    </div>
                                  )}
                                </div>
                              </div>

                              {/* Step 3: Context Retrieval */}
                              <div className="flex items-start gap-3">
                                <div className="w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                                  <span className="text-xs text-purple-300">
                                    3
                                  </span>
                                </div>
                                <div className="flex-1">
                                  <div className="text-xs font-semibold text-purple-300 mb-1">
                                    Context Retrieval
                                  </div>
                                  <div className="text-xs text-gray-400">
                                    Extracting relevant chunks from documents
                                  </div>
                                </div>
                              </div>

                              {/* Step 4: LLM Response */}
                              <div className="flex items-start gap-3">
                                <div className="w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                                  <span className="text-xs text-purple-300">
                                    4
                                  </span>
                                </div>
                                <div className="flex-1">
                                  <div className="text-xs font-semibold text-purple-300 mb-1">
                                    LLM Generation
                                  </div>
                                  <div className="text-xs text-gray-400">
                                    Generating response using{" "}
                                    <span className="text-purple-400">
                                      {settings.model === "auto"
                                        ? "Auto-selected model"
                                        : settings.model}
                                    </span>{" "}
                                    with retrieved context
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Other Tools */}
                        {msg.tools
                          ?.filter((t) => t.type !== "rag")
                          .map((tool, i) => (
                            <div
                              key={i}
                              className="mt-3 p-3 bg-black/30 rounded-lg text-sm"
                            >
                              {tool.type === "tool" && (
                                <div className="flex items-start gap-2">
                                  <Sparkles className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                                      <span className="font-mono font-semibold">
                                        {tool.name}
                                      </span>
                                      <span className="text-gray-400">→</span>
                                      <span className="text-green-400">
                                        executed
                                      </span>
                                    </div>
                                    {tool.result && (
                                      <div className="mt-2 overflow-x-auto custom-scrollbar">
                                        <pre className="text-xs font-mono bg-black/40 p-2 rounded whitespace-pre min-w-max">
                                          {(() => {
                                            try {
                                              const parsed = JSON.parse(
                                                tool.result
                                              );
                                              return JSON.stringify(
                                                parsed,
                                                null,
                                                2
                                              );
                                            } catch {
                                              return tool.result;
                                            }
                                          })()}
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                      </div>

                      {msg.role === "user" && (
                        <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
                          <User className="w-5 h-5" />
                        </div>
                      )}
                    </div>
                  ))}

                {activeThread?.messages.filter(
                  (msg) => msg.messageMode === "rag"
                ).length === 0 && (
                  <div className="text-center text-gray-400 mt-8">
                    <div className="mb-4">
                      <FileText className="w-16 h-16 mx-auto text-purple-500/50 mb-4" />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">
                      RAG Knowledge Base Assistant
                    </h3>
                    <p className="text-sm mb-4">
                      Upload documents to build your knowledge base, then ask
                      questions!
                    </p>
                    <div className="text-xs text-gray-500 space-y-1">
                      <p>• Upload PDF, TXT, MD, DOC files</p>
                      <p>• Documents are indexed using OpenAI Embeddings</p>
                      <p>• Ask questions based on your documents</p>
                    </div>
                  </div>
                )}

                {/* Loading indicator */}
                {loading &&
                  !activeThread?.messages.some((msg) => msg.isStreaming) && (
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                        <Bot className="w-5 h-5" />
                      </div>
                      <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-4">
                        <div className="flex gap-2">
                          <div className="w-2 h-2 bg-white rounded-full animate-bounce" />
                          <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-100" />
                          <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-200" />
                        </div>
                      </div>
                    </div>
                  )}

                <div ref={chatEndRef} />
              </div>

              {/* Input */}
              <div className="flex-shrink-0 p-6 bg-black/30 backdrop-blur-xl border-t border-white/10">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Ask a question based on your documents..."
                    className="flex-1 bg-white/5 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                  <button
                    onClick={handleSend}
                    disabled={loading || ragProcessing}
                    className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg px-6 py-3 transition flex items-center gap-2"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
                {ragFiles.length === 0 && (
                  <p className="text-xs text-gray-400 mt-2 text-center">
                    💡 Upload documents first to enable RAG functionality
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* PHOTO MODE UI */}
        {mode === "photo" && (
          <div className="flex-1 flex flex-col min-h-0">
            {/* Photo Sub-Mode Tabs */}
            <div className="flex gap-2 p-4 border-b border-white/10">
              <button
                onClick={() => {
                  setPhotoSubMode("generate");
                  setShowGallery(false);
                }}
                className={`px-4 py-2 rounded-lg transition ${
                  photoSubMode === "generate" && !showGallery
                    ? "bg-purple-600"
                    : "bg-white/5 hover:bg-white/10"
                }`}
              >
                🎨 Generate
              </button>
              <button
                onClick={() => {
                  setPhotoSubMode("analyze");
                  setShowGallery(false);
                }}
                className={`px-4 py-2 rounded-lg transition ${
                  photoSubMode === "analyze" && !showGallery
                    ? "bg-purple-600"
                    : "bg-white/5 hover:bg-white/10"
                }`}
              >
                🔍 Analyze
              </button>
              <button
                onClick={() => {
                  setShowGallery(!showGallery);
                  if (!showGallery) {
                    loadGallery();
                  } else {
                    setPhotoSubMode("generate"); // Скинути на generate при закритті галереї
                  }
                }}
                className={`px-4 py-2 rounded-lg transition ${
                  showGallery ? "bg-purple-600" : "bg-white/5 hover:bg-white/10"
                }`}
              >
                🖼️ Gallery ({gallery.length})
              </button>
            </div>

            {/* Gallery View */}
            {showGallery ? (
              <div className="flex-1 flex flex-col min-h-0">
                {/* Gallery Grid */}
                <div className="flex-1 overflow-y-auto p-6 min-h-0">
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {gallery.map((item) => (
                      <div
                        key={item.id}
                        className="bg-white/5 rounded-lg overflow-hidden hover:bg-white/10 transition cursor-pointer group"
                        onClick={() => setSelectedImage(item.image_url)}
                      >
                        <div className="relative">
                          <img
                            src={item.image_url}
                            alt={item.prompt}
                            className="w-full aspect-square object-cover"
                          />
                          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition flex items-center justify-center">
                            <span className="text-white opacity-0 group-hover:opacity-100 text-sm font-semibold">
                              Click to view
                            </span>
                          </div>
                        </div>
                        <div className="p-3">
                          <p className="text-sm text-gray-300 line-clamp-2 mb-2">
                            {item.prompt}
                          </p>
                          <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                            <span>{item.model}</span>
                            <span>{item.size}</span>
                          </div>
                          <button
                            onClick={async (e) => {
                              e.stopPropagation();
                              try {
                                const response = await fetch(
                                  `${API_BASE_URL}/gallery/${item.id}`,
                                  { method: "DELETE" }
                                );
                                if (response.ok) {
                                  loadGallery();
                                }
                              } catch (error) {
                                console.error("Error deleting image:", error);
                              }
                            }}
                            className="w-full text-red-400 hover:text-red-300 text-xs py-1 hover:bg-red-400/10 rounded transition"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                  {gallery.length === 0 && (
                    <div className="text-center text-gray-400 mt-8">
                      No images in gallery yet. Generate some images first!
                    </div>
                  )}
                </div>
              </div>
            ) : photoSubMode === "generate" ? (
              /* Generate View with Settings */
              <div className="flex-1 flex flex-col min-h-0">
                {/* Settings Panel for Generate */}
                <div className="flex-shrink-0 p-4 bg-black/20 border-b border-white/10">
                  <h3 className="text-sm font-semibold mb-3">
                    Image Generation Settings (DALL-E)
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div>
                      <label className="block text-xs mb-1 text-gray-400">
                        Model
                      </label>
                      <select
                        value={settings.imageSettings?.model || "dall-e-3"}
                        onChange={(e) =>
                          setSettings({
                            ...settings,
                            imageSettings: {
                              ...settings.imageSettings,
                              model: e.target.value,
                            },
                          })
                        }
                        className="w-full bg-slate-800 rounded-lg p-2 text-sm text-white border border-purple-500/30 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 custom-select"
                      >
                        <option
                          value="dall-e-3"
                          className="bg-slate-800 text-white"
                        >
                          DALL-E 3
                        </option>
                        <option
                          value="dall-e-2"
                          className="bg-slate-800 text-white"
                        >
                          DALL-E 2
                        </option>
                      </select>
                    </div>

                    {settings.imageSettings?.model === "dall-e-3" ? (
                      <>
                        <div>
                          <label className="block text-xs mb-1 text-gray-400">
                            Size
                          </label>
                          <select
                            value={settings.imageSettings?.size || "1024x1024"}
                            onChange={(e) =>
                              setSettings({
                                ...settings,
                                imageSettings: {
                                  ...settings.imageSettings,
                                  size: e.target.value,
                                },
                              })
                            }
                            className="w-full bg-slate-800 rounded-lg p-2 text-sm text-white border border-purple-500/30 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 custom-select"
                          >
                            <option
                              value="1024x1024"
                              className="bg-slate-800 text-white"
                            >
                              1024x1024
                            </option>
                            <option
                              value="1792x1024"
                              className="bg-slate-800 text-white"
                            >
                              1792x1024
                            </option>
                            <option
                              value="1024x1792"
                              className="bg-slate-800 text-white"
                            >
                              1024x1792
                            </option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs mb-1 text-gray-400">
                            Quality
                          </label>
                          <select
                            value={
                              settings.imageSettings?.quality || "standard"
                            }
                            onChange={(e) =>
                              setSettings({
                                ...settings,
                                imageSettings: {
                                  ...settings.imageSettings,
                                  quality: e.target.value,
                                },
                              })
                            }
                            className="w-full bg-slate-800 rounded-lg p-2 text-sm text-white border border-purple-500/30 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 custom-select"
                          >
                            <option
                              value="standard"
                              className="bg-slate-800 text-white"
                            >
                              Standard
                            </option>
                            <option
                              value="hd"
                              className="bg-slate-800 text-white"
                            >
                              HD
                            </option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs mb-1 text-gray-400">
                            Style
                          </label>
                          <select
                            value={settings.imageSettings?.style || "vivid"}
                            onChange={(e) =>
                              setSettings({
                                ...settings,
                                imageSettings: {
                                  ...settings.imageSettings,
                                  style: e.target.value,
                                },
                              })
                            }
                            className="w-full bg-slate-800 rounded-lg p-2 text-sm text-white border border-purple-500/30 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 custom-select"
                          >
                            <option
                              value="vivid"
                              className="bg-slate-800 text-white"
                            >
                              Vivid
                            </option>
                            <option
                              value="natural"
                              className="bg-slate-800 text-white"
                            >
                              Natural
                            </option>
                          </select>
                        </div>
                      </>
                    ) : (
                      <div>
                        <label className="block text-xs mb-1 text-gray-400">
                          Size
                        </label>
                        <select
                          value={settings.imageSettings?.size || "1024x1024"}
                          onChange={(e) =>
                            setSettings({
                              ...settings,
                              imageSettings: {
                                ...settings.imageSettings,
                                size: e.target.value,
                              },
                            })
                          }
                          className="w-full bg-slate-800 rounded-lg p-2 text-sm text-white border border-purple-500/30 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 custom-select"
                        >
                          <option
                            value="256x256"
                            className="bg-slate-800 text-white"
                          >
                            256x256
                          </option>
                          <option
                            value="512x512"
                            className="bg-slate-800 text-white"
                          >
                            512x512
                          </option>
                          <option
                            value="1024x1024"
                            className="bg-slate-800 text-white"
                          >
                            1024x1024
                          </option>
                        </select>
                      </div>
                    )}
                  </div>
                </div>

                {/* Messages for Generate */}
                <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">
                  {activeThread?.messages
                    .filter((msg) => {
                      const msgMode =
                        msg.messageMode ||
                        (msg.image
                          ? "image-analyze"
                          : msg.image_url
                          ? "image-gen"
                          : null);
                      if (photoSubMode === "generate") {
                        // Показувати тільки повідомлення, пов'язані з генерацією
                        return (
                          msgMode === "image-gen" ||
                          (msg.role === "user" && msgMode === "image-gen") ||
                          msg.tools?.some((t) => t.type === "image") ||
                          msg.image_url
                        );
                      } else if (photoSubMode === "analyze") {
                        // Показувати тільки повідомлення, пов'язані з аналізом
                        return (
                          msgMode === "image-analyze" ||
                          (msg.role === "user" &&
                            (msgMode === "image-analyze" || msg.image)) ||
                          (msg.role === "assistant" &&
                            msgMode === "image-analyze")
                        );
                      }
                      return false;
                    })
                    .map((msg, idx) => (
                      <div
                        key={idx}
                        className={`flex gap-3 ${
                          msg.role === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        {msg.role === "assistant" && (
                          <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                            <Bot className="w-5 h-5" />
                          </div>
                        )}

                        <div
                          className={`max-w-2xl rounded-2xl p-4 ${
                            msg.role === "user"
                              ? "bg-purple-600"
                              : "bg-white/10 backdrop-blur-xl"
                          }`}
                        >
                          {msg.image && (
                            <img
                              src={msg.image}
                              alt="uploaded"
                              className="rounded-lg mb-2 max-w-md"
                            />
                          )}
                          {msg.image_url && (
                            <img
                              src={msg.image_url}
                              alt="generated"
                              className="rounded-lg mb-2 max-w-md"
                            />
                          )}
                          <div className="whitespace-pre-wrap">
                            {formatMessageContent(msg.content)}
                          </div>
                        </div>

                        {msg.role === "user" && (
                          <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
                            <User className="w-5 h-5" />
                          </div>
                        )}
                      </div>
                    ))}

                  {activeThread?.messages.filter((msg) => {
                    const msgMode =
                      msg.messageMode ||
                      (msg.image
                        ? "image-analyze"
                        : msg.image_url
                        ? "image-gen"
                        : null);
                    if (photoSubMode === "generate") {
                      return (
                        msgMode === "image-gen" ||
                        (msg.role === "user" && msgMode === "image-gen") ||
                        msg.tools?.some((t) => t.type === "image") ||
                        msg.image_url
                      );
                    } else if (photoSubMode === "analyze") {
                      return (
                        msgMode === "image-analyze" ||
                        (msg.role === "user" &&
                          (msgMode === "image-analyze" || msg.image)) ||
                        (msg.role === "assistant" &&
                          msgMode === "image-analyze")
                      );
                    }
                    return false;
                  }).length === 0 && (
                    <div className="text-center text-gray-400 mt-8">
                      {photoSubMode === "generate"
                        ? "Generate your first image using DALL-E!"
                        : "Upload an image to analyze it with GPT-4 Vision"}
                    </div>
                  )}

                  {/* Loading indicator for Photo mode */}
                  {loading &&
                    !activeThread?.messages.some((msg) => msg.isStreaming) && (
                      <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                          <Bot className="w-5 h-5" />
                        </div>
                        <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-4">
                          <div className="flex gap-2">
                            <div className="w-2 h-2 bg-white rounded-full animate-bounce" />
                            <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-100" />
                            <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-200" />
                          </div>
                        </div>
                      </div>
                    )}

                  <div ref={chatEndRef} />
                </div>
              </div>
            ) : (
              /* Analyze View */
              <div className="flex-1 flex flex-col min-h-0">
                <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">
                  {activeThread?.messages
                    .filter((msg) => {
                      const msgMode =
                        msg.messageMode ||
                        (msg.image
                          ? "image-analyze"
                          : msg.image_url
                          ? "image-gen"
                          : null);
                      if (photoSubMode === "analyze") {
                        return (
                          msgMode === "image-analyze" ||
                          (msg.role === "user" &&
                            (msgMode === "image-analyze" || msg.image)) ||
                          (msg.role === "assistant" &&
                            msgMode === "image-analyze")
                        );
                      }
                      return false;
                    })
                    .map((msg, idx) => (
                      <div
                        key={idx}
                        className={`flex gap-3 ${
                          msg.role === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        {msg.role === "assistant" && (
                          <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                            <Bot className="w-5 h-5" />
                          </div>
                        )}

                        <div
                          className={`max-w-2xl rounded-2xl p-4 ${
                            msg.role === "user"
                              ? "bg-purple-600"
                              : "bg-white/10 backdrop-blur-xl"
                          }`}
                        >
                          {msg.image && (
                            <img
                              src={msg.image}
                              alt="uploaded"
                              className="rounded-lg mb-2 max-w-md cursor-pointer"
                              onClick={() => setSelectedImage(msg.image)}
                            />
                          )}
                          {msg.image_url && (
                            <img
                              src={msg.image_url}
                              alt="generated"
                              className="rounded-lg mb-2 max-w-md"
                            />
                          )}
                          <div className="whitespace-pre-wrap">
                            {formatMessageContent(msg.content)}
                          </div>
                        </div>

                        {msg.role === "user" && (
                          <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
                            <User className="w-5 h-5" />
                          </div>
                        )}
                      </div>
                    ))}

                  {activeThread?.messages.filter((msg) => {
                    const msgMode =
                      msg.messageMode ||
                      (msg.image
                        ? "image-analyze"
                        : msg.image_url
                        ? "image-gen"
                        : null);
                    if (photoSubMode === "analyze") {
                      return (
                        msgMode === "image-analyze" ||
                        (msg.role === "user" &&
                          (msgMode === "image-analyze" || msg.image)) ||
                        (msg.role === "assistant" &&
                          msgMode === "image-analyze")
                      );
                    }
                    return false;
                  }).length === 0 && (
                    <div className="text-center text-gray-400 mt-8">
                      Upload an image to analyze it with GPT-4 Vision
                    </div>
                  )}

                  {/* Loading indicator for Analyze */}
                  {loading &&
                    !activeThread?.messages.some((msg) => msg.isStreaming) && (
                      <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                          <Bot className="w-5 h-5" />
                        </div>
                        <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-4">
                          <div className="flex gap-2">
                            <div className="w-2 h-2 bg-white rounded-full animate-bounce" />
                            <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-100" />
                            <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-200" />
                          </div>
                        </div>
                      </div>
                    )}

                  <div ref={chatEndRef} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* MESSAGES (Chat Mode) */}
        {mode !== "photo" && mode !== "rag" && (
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {activeThread?.messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-3 ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {msg.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5" />
                  </div>
                )}

                <div
                  className={`max-w-2xl rounded-2xl p-4 ${
                    msg.role === "user"
                      ? "bg-purple-600"
                      : "bg-white/10 backdrop-blur-xl"
                  }`}
                >
                  {msg.image && (
                    <img
                      src={msg.image}
                      alt="uploaded"
                      className="rounded-lg mb-2 max-w-md"
                    />
                  )}
                  {msg.image_url && (
                    <img
                      src={msg.image_url}
                      alt="generated"
                      className="rounded-lg mb-2 max-w-md"
                    />
                  )}
                  <div className="whitespace-pre-wrap">
                    {formatMessageContent(msg.content)}
                    {msg.isStreaming && (
                      <span className="inline-block w-0.5 h-5 bg-purple-400 ml-1.5 align-middle streaming-cursor" />
                    )}
                  </div>

                  {msg.tools?.map((tool, i) => (
                    <div
                      key={i}
                      className="mt-3 p-3 bg-black/30 rounded-lg text-sm"
                    >
                      {tool.type === "image" && (
                        <img
                          src={tool.url}
                          alt="generated"
                          className="rounded-lg w-full"
                        />
                      )}
                      {tool.type === "tool" && (
                        <div className="flex items-start gap-2">
                          <Sparkles className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                              <span className="font-mono font-semibold">
                                {tool.name}
                              </span>
                              <span className="text-gray-400">→</span>
                              <span className="text-green-400">executed</span>
                            </div>
                            {tool.result && (
                              <div className="mt-2 overflow-x-auto custom-scrollbar">
                                <pre className="text-xs font-mono bg-black/40 p-2 rounded whitespace-pre min-w-max">
                                  {(() => {
                                    try {
                                      const parsed = JSON.parse(tool.result);
                                      return JSON.stringify(parsed, null, 2);
                                    } catch {
                                      return tool.result;
                                    }
                                  })()}
                                </pre>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      {tool.type === "rag" && (
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-blue-400" />
                          <span>Sources: {tool.docs.join(", ")}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {msg.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5" />
                  </div>
                )}
              </div>
            ))}

            {loading &&
              !activeThread?.messages.some((msg) => msg.isStreaming) && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5" />
                  </div>
                  <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-4">
                    <div className="flex gap-2">
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-100" />
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-200" />
                    </div>
                  </div>
                </div>
              )}

            <div ref={chatEndRef} />
          </div>
        )}

        {/* INPUT - Hide for Gallery and RAG (RAG has its own input) */}
        {!(mode === "photo" && showGallery) && mode !== "rag" && (
          <div className="p-6 bg-black/30 backdrop-blur-xl border-t border-white/10">
            {imageFile && (
              <div className="mb-3 flex items-center gap-2 p-2 bg-white/5 rounded-lg">
                <Image className="w-4 h-4 text-purple-400" />
                <span className="text-sm flex-1">{imageFile.name}</span>
                <button
                  onClick={() => setImageFile(null)}
                  className="text-red-400 hover:text-red-300"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            <div className="flex gap-3">
              {(mode === "photo" && photoSubMode === "analyze") ||
              mode === "image-analyze" ||
              mode === "chat" ? (
                <label className="cursor-pointer bg-white/5 hover:bg-white/10 rounded-lg p-3 transition">
                  <Upload className="w-5 h-5" />
                  <input
                    type="file"
                    className="hidden"
                    accept="image/*"
                    onChange={(e) => setImageFile(e.target.files[0])}
                  />
                </label>
              ) : null}

              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSend()}
                placeholder={
                  mode === "photo"
                    ? photoSubMode === "generate"
                      ? "Describe image to generate with DALL-E..."
                      : "Ask a question about the image..."
                    : mode === "image-gen"
                    ? "Describe image to generate..."
                    : "Type your message..."
                }
                className="flex-1 bg-white/5 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />

              <button
                onClick={handleSend}
                disabled={loading}
                className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg px-6 py-3 transition flex items-center gap-2"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}

        {/* Image Modal */}
        {selectedImage && (
          <div
            className="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedImage(null)}
          >
            <div className="relative max-w-7xl max-h-full">
              <button
                onClick={() => setSelectedImage(null)}
                className="absolute top-4 right-4 bg-black/50 hover:bg-black/70 rounded-full p-2 transition z-10"
              >
                <X className="w-6 h-6 text-white" />
              </button>
              <img
                src={selectedImage}
                alt="Full size"
                className="max-w-full max-h-[90vh] object-contain rounded-lg"
                onClick={(e) => e.stopPropagation()}
              />
            </div>
          </div>
        )}
      </div>

      {/* SETTINGS PANEL */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-900 rounded-2xl p-6 w-96 border border-white/10">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Settings</h3>
              <button onClick={() => setShowSettings(false)}>
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm mb-2">Model</label>
                <select
                  value={settings.model}
                  onChange={(e) =>
                    setSettings({ ...settings, model: e.target.value })
                  }
                  className="w-full bg-slate-800 rounded-lg p-2 text-white border border-purple-500/30 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 custom-select"
                >
                  <option
                    value="gpt-4o-mini"
                    className="bg-slate-800 text-white"
                  >
                    GPT-4o Mini (Швидкий, економний)
                  </option>
                  <option value="gpt-4o" className="bg-slate-800 text-white">
                    GPT-4o (Найкраща якість, складні задачі)
                  </option>
                  <option value="auto" className="bg-slate-800 text-white">
                    Auto (Автоматичний вибір)
                  </option>
                </select>
                <p className="text-xs text-gray-400 mt-1">
                  {settings.model === "auto"
                    ? "Система автоматично вибере модель на основі складності запиту"
                    : settings.model === "gpt-4o-mini"
                    ? "Рекомендовано для простих запитів та економії"
                    : "Рекомендовано для складних задач та аналізу"}
                </p>
              </div>

              <div>
                <label className="block text-sm mb-2">
                  Temperature: {settings.temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      temperature: parseFloat(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.enableRAG}
                    onChange={(e) =>
                      setSettings({ ...settings, enableRAG: e.target.checked })
                    }
                    className="rounded"
                  />
                  <span className="text-sm">Enable RAG (File Search)</span>
                </label>

                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.enableAgent}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        enableAgent: e.target.checked,
                      })
                    }
                    className="rounded"
                  />
                  <span className="text-sm">Enable Agent Tools</span>
                </label>

                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.enableStreaming ?? true}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        enableStreaming: e.target.checked,
                      })
                    }
                    className="rounded"
                    disabled={!settings.enableAgent || mode !== "chat"}
                  />
                  <span className="text-sm">
                    Enable Streaming (Real-time responses)
                  </span>
                </label>
                {(!settings.enableAgent || mode !== "chat") && (
                  <p className="text-xs text-gray-400 ml-6">
                    Streaming доступний тільки для chat mode з увімкненим Agent
                  </p>
                )}
              </div>

              {/* Image Analysis Settings */}
              {(mode === "photo" || mode === "image-analyze") && (
                <div className="pt-4 border-t border-white/10">
                  <h4 className="text-sm font-semibold mb-3">
                    Image Analysis (VQA)
                  </h4>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.detailedAnalysis ?? true}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          detailedAnalysis: e.target.checked,
                        })
                      }
                      className="rounded"
                    />
                    <span className="text-sm">
                      Detailed Analysis (High detail mode)
                    </span>
                  </label>
                  <p className="text-xs text-gray-400 mt-1 ml-6">
                    Використовує GPT-4o Vision для детального аналізу зображень
                  </p>
                </div>
              )}

              <div className="pt-4 border-t border-white/10">
                <h4 className="text-sm font-semibold mb-3">Agent Tools</h4>
                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.enabledTools?.get_item_price ?? true}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          enabledTools: {
                            ...settings.enabledTools,
                            get_item_price: e.target.checked,
                          },
                        })
                      }
                      className="rounded"
                    />
                    <span className="text-sm">💰 Get Item Price</span>
                  </label>

                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={
                        settings.enabledTools?.calculate_shipping ?? true
                      }
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          enabledTools: {
                            ...settings.enabledTools,
                            calculate_shipping: e.target.checked,
                          },
                        })
                      }
                      className="rounded"
                    />
                    <span className="text-sm">🚚 Calculate Shipping</span>
                  </label>

                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.enabledTools?.book_meeting ?? true}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          enabledTools: {
                            ...settings.enabledTools,
                            book_meeting: e.target.checked,
                          },
                        })
                      }
                      className="rounded"
                    />
                    <span className="text-sm">
                      📅 Book Meeting (Google Calendar)
                    </span>
                  </label>

                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.enabledTools?.send_email ?? true}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          enabledTools: {
                            ...settings.enabledTools,
                            send_email: e.target.checked,
                          },
                        })
                      }
                      className="rounded"
                    />
                    <span className="text-sm">✉️ Send Email (Gmail)</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MultimodalAIService;
