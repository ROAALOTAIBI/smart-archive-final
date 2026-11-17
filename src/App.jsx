
import "./App.css";
import { createEmbedding, cosineSimilarity } from "./semanticSearch.js";
import { useEffect, useMemo, useState } from "react";
import { supabase } from "./supabaseClient";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import mammoth from "mammoth";
import { createWorker } from "tesseract.js";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

  
async function extractText(file) {
  const extension = file.name.split(".").pop().toLowerCase();

  // TXT
  if (extension === "txt") {
    return (await file.text()).trim();
  }

  // DOCX
  if (extension === "docx") {
    const buffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({
      arrayBuffer: buffer,
    });

    return result.value.trim();
  }

  // PDF: extract native text first, OCR only scanned pages
  if (extension === "pdf") {
  const buffer = await file.arrayBuffer();

  const pdf = await pdfjsLib.getDocument({
    data: buffer,
  }).promise;

  const pages = [];
  const worker = await createWorker("ara+eng");

  try {
    await worker.setParameters({
      preserve_interword_spaces: "1",
      user_defined_dpi: "300",
    });

    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
      const page = await pdf.getPage(pageNumber);
      const viewport = page.getViewport({ scale: 5 });

      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");

      canvas.width = Math.ceil(viewport.width);
      canvas.height = Math.ceil(viewport.height);

      context.fillStyle = "#FFFFFF";
      context.fillRect(0, 0, canvas.width, canvas.height);

      await page.render({
        canvasContext: context,
        viewport,
      }).promise;

      const imageData = context.getImageData(
        0, 0, canvas.width, canvas.height
      );

      const pixels = imageData.data;

      for (let i = 0; i < pixels.length; i += 4) {
        const gray =
          0.299 * pixels[i] +
          0.587 * pixels[i + 1] +
          0.114 * pixels[i + 2];

        const contrast = Math.max(
          0,
          Math.min(255, (gray - 128) * 1.3 + 128)
        );

        pixels[i] = contrast;
        pixels[i + 1] = contrast;
        pixels[i + 2] = contrast;
      }

      context.putImageData(imageData, 0, 0);

      const result = await worker.recognize(canvas);
      pages.push((result.data.text || "").trim());

      canvas.width = 0;
      canvas.height = 0;
      page.cleanup();
    }

    return pages.filter(Boolean).join("\n\n").trim();
   } finally {
  await worker.terminate();
}
}

  // Images: PNG, JPG, WEBP, BMP, TIFF
  if (file.type.startsWith("image/")) {
    const worker = await createWorker("ara+eng");

    try {
      const image = await createImageBitmap(file);
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");

      if (!context) {
        image.close();
        throw new Error("تعذر تجهيز الصورة للقراءة.");
      }

      const scale = Math.min(
        2.5,
        4000 / Math.max(image.width, image.height)
      );

      canvas.width = Math.round(image.width * scale);
      canvas.height = Math.round(image.height * scale);

      context.fillStyle = "#FFFFFF";
      context.fillRect(0, 0, canvas.width, canvas.height);

      context.drawImage(
        image,
        0,
        0,
        canvas.width,
        canvas.height
      );

      const result = await worker.recognize(canvas);

      image.close();
      canvas.width = 0;
      canvas.height = 0;

      return (result.data.text || "").trim();
    } finally {
      await worker.terminate();
    }
  }

  throw new Error("نوع الملف غير مدعوم.");
}
async function createImagePreviewUrl(storagePath) {
  if (!storagePath) return null;

  const { data, error } = await supabase.storage
    .from("archive-files")
    .createSignedUrl(storagePath, 3600);

  if (error) {
    console.error("تعذر إنشاء رابط الصورة:", error);
    return null;
  }

  return data?.signedUrl || null;
}
function highlightText(text, query) {
  const words = [...new Set(query.trim().split(/\s+/).filter(Boolean))];

  if (words.length === 0) return text;

  const escapedWords = words.map((word) =>
    word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  );

  const regex = new RegExp(`(${escapedWords.join("|")})`, "gi");

  return text.split(regex).map((part, index) =>
    words.some((word) => part.toLowerCase() === word.toLowerCase()) ? (
      <mark
        key={index}
        style={{
          backgroundColor: "#FFE58A",
          color: "#172C3D",
          padding: "2px 4px",
          borderRadius: "4px",
          fontWeight: "bold",
        }}
      >
        {part}
      </mark>
    ) : (
      part
    )
  );
}

function getSearchSnippet(text, query) {
  const source = (text || "").replace(/\s+/g, " ").trim();

  if (!source) return "";

  const searchTerm = (query || "").trim().toLowerCase();

  if (!searchTerm) return source.slice(0, 300);

  const sentences = source
    .split(/(?<=[.!?؟؛])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  const exactIndex = sentences.findIndex((sentence) =>
    sentence.toLowerCase().includes(searchTerm)
  );

  const words = searchTerm.split(/\s+/).filter(Boolean);

  const matchIndex =
    exactIndex !== -1
      ? exactIndex
      : sentences.findIndex((sentence) =>
          words.some((word) => sentence.toLowerCase().includes(word))
        );

  if (matchIndex === -1) return "";

  return sentences
    .slice(
      Math.max(0, matchIndex - 1),
      Math.min(sentences.length, matchIndex + 2)
    )
    .join(" ");
}

function createTextChunks(files) {
  const chunks = [];
  const chunkSize = 1000;
  const step = 800;

  for (const file of files) {
    const text = file.text || "";

    for (let start = 0; start < text.length; start += step) {
      const snippet = text.slice(start, start + chunkSize).trim();

      if (!snippet) continue;

      chunks.push({
        fileName: file.name,
        snippet,
      });
    }
  }

  return chunks;
}

function App() {
  // Archive states
  const [files, setFiles] = useState([]);
  const [filesLoaded, setFilesLoaded] = useState(false);

  const [query, setQuery] = useState("");
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [semanticResults, setSemanticResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  // Authentication states
  const [session, setSession] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authMode, setAuthMode] = useState("login");
  const [authMessage, setAuthMessage] = useState("");
  const [authBusy, setAuthBusy] = useState(false);

  // Check current login session
  useEffect(() => {
    let mounted = true;

    supabase.auth.getSession().then(({ data, error }) => {
      if (!mounted) return;

      if (error) {
        setAuthMessage(error.message);
      }

      setSession(data?.session ?? null);
      setAuthLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, currentSession) => {
      if (!mounted) return;

      setSession(currentSession);
      setAuthLoading(false);
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  // Load only the logged-in user's files from Supabase
  useEffect(() => {
    let cancelled = false;

    async function loadUserFiles() {
      const userId = session?.user?.id;

      if (!userId) {
        setFiles([]);
        setFilesLoaded(true);
        return;
      }

      setFilesLoaded(false);
      setStatus("جارٍ تحميل ملفاتك من الأرشيف...");

      try {
        const { data, error } = await supabase
          .from("archive_files")
          .select(
            "id, user_id, name, content, file_size, mime_type, storage_path, embedding, created_at"
          )
          .eq("user_id", userId)
          .order("created_at", { ascending: false });

        if (error) throw error;

        if (cancelled) return;

     const loadedFiles = await Promise.all(
  (data || []).map(async (row) => ({
    id: row.id,
    name: row.name,
    size: row.file_size || 0,
    type: row.mime_type || "",
    text: row.content || "",
    embedding: Array.isArray(row.embedding)
      ? row.embedding
      : null,
    storagePath: row.storage_path || null,
    imageUrl: row.mime_type?.startsWith("image/")
      ? await createImagePreviewUrl(row.storage_path)
      : null,
    pdfUrl: null,
    error: "",
  }))
);

        setFiles(loadedFiles);
        setStatus("تم تحميل أرشيفك بنجاح.");
      } catch (error) {
        console.error("تعذر تحميل الملفات:", error);

        if (!cancelled) {
          setFiles([]);
          setStatus("تعذر تحميل ملفات الأرشيف: " + error.message);
        }
      } finally {
        if (!cancelled) {
          setFilesLoaded(true);
        }
      }
    }

    loadUserFiles();

    return () => {
      cancelled = true;
    };
  }, [session?.user?.id]);

  // Sign up or sign in
  const handleAuth = async (event) => {
    event.preventDefault();

    if (!email.trim() || !password) {
      setAuthMessage("يرجى إدخال البريد الإلكتروني وكلمة المرور.");
      return;
    }

    if (password.length < 6) {
      setAuthMessage("يجب أن تتكون كلمة المرور من 6 أحرف على الأقل.");
      return;
    }

    setAuthBusy(true);
    setAuthMessage("");

    try {
      if (authMode === "signup") {
        const { data, error } = await supabase.auth.signUp({
          email: email.trim(),
          password,
        });

        if (error) throw error;

        if (data.session) {
          setAuthMessage("تم إنشاء الحساب بنجاح.");
        } else {
          setAuthMessage(
            "تم إنشاء الحساب. تحققي من بريدك الإلكتروني إذا طُلب منك تأكيده."
          );
        }
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email: email.trim(),
          password,
        });

        if (error) throw error;
      }
    } catch (error) {
      setAuthMessage(
        error.message || "حدث خطأ أثناء تسجيل الدخول. حاولي مرة أخرى."
      );
    } finally {
      setAuthBusy(false);
    }
  };

  // Sign out
  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();

    if (error) {
      setAuthMessage(error.message);
    } else {
      setPassword("");
      setEmail("");
      setAuthMessage("");
      setAiAnswer(null);
      setAiQuestion("");
      setQuery("");
    }
  };

  // Lexical search
  const lexicalFiltered = useMemo(() => {
    const searchTerm = query.trim().toLowerCase();

    if (!searchTerm) return files;

    const searchWords = [...new Set(searchTerm.split(/\s+/))];

    return files
      .map((file) => {
        const name = file.name.toLowerCase();
        const text = (file.text || "").toLowerCase();
        let score = 0;

        for (const word of searchWords) {
          if (name.includes(word)) score += 10;

          let position = 0;

          while ((position = text.indexOf(word, position)) !== -1) {
            score += 1;
            position += word.length;
          }
        }

        return { ...file, score };
      })
      .filter((file) => file.score > 0)
      .sort((a, b) => b.score - a.score);
  }, [files, query]);

  // Semantic search
  useEffect(() => {
    let cancelled = false;
    const searchTerm = query.trim();

    if (!searchTerm) {
      setSemanticResults(null);
      return;
    }

    const hasEmbeddings = files.some(
      (file) => Array.isArray(file.embedding) && file.embedding.length > 0
    );

    if (!hasEmbeddings) {
      setSemanticResults(null);
      return;
    }

    async function runSemanticSearch() {
      try {
        const queryEmbedding = await createEmbedding(searchTerm);

        const results = files
          .map((file) => {
            const lexicalScore =
              lexicalFiltered.find((item) => item.id === file.id)?.score || 0;

            const semanticScore =
              Array.isArray(file.embedding) && file.embedding.length > 0
                ? cosineSimilarity(queryEmbedding, file.embedding)
                : 0;

            return {
              ...file,
              score: lexicalScore + semanticScore * 10,
              semanticScore,
            };
          })
          .filter(
            (file) =>
              Array.isArray(file.embedding) &&
              file.embedding.length > 0 &&
              file.semanticScore >= 0.3
          )
          .sort((a, b) => {
            if (b.semanticScore !== a.semanticScore) {
              return b.semanticScore - a.semanticScore;
            }

            return b.score - a.score;
          });

        if (!cancelled) {
          setSemanticResults(results);

          const bestScore = results.length
            ? results[0].semanticScore.toFixed(3)
            : "لا توجد نتائج";

          setStatus(
            `اكتمل البحث الدلالي. عدد النتائج: ${results.length}، أعلى تشابه: ${bestScore}`
          );
        }
      } catch (error) {
        console.error("تعذر البحث الدلالي:", error);

        if (!cancelled) {
          setSemanticResults(null);
          setStatus("تعذر البحث الدلالي. راجع رسالة الخطأ في الكود.");
        }
      }
    }

    runSemanticSearch();

    return () => {
      cancelled = true;
    };
  }, [query, files, lexicalFiltered]);

  const filtered = useMemo(() => {
    if (!query.trim()) return files;

    const combined = [...(semanticResults ?? [])];

    lexicalFiltered.forEach((file) => {
      if (!combined.some((item) => item.id === file.id)) {
        combined.push(file);
      }
    });

    return combined;
  }, [query, files, semanticResults, lexicalFiltered]);

  // Upload files to Supabase Storage and save their data in Supabase Database
  const handleUpload = async (event) => {
    const selectedFiles = Array.from(event.target.files || []);
    const userId = session?.user?.id;

    if (selectedFiles.length === 0) return;

    if (!userId) {
      setStatus("يرجى تسجيل الدخول قبل رفع الملفات.");
      event.target.value = "";
      return;
    }

    setLoading(true);
    setStatus("جارٍ استخراج النصوص وحفظ الملفات...");

    const processedFiles = [];
    let successCount = 0;

    for (const file of selectedFiles) {
      let storagePath = null;

      try {
        // Extract file text
        const text = await extractText(file);

        // Create embedding
        let embedding = null;

        if (text && text.trim()) {
          try {
            embedding = await createEmbedding(text.slice(0, 4000));
          } catch (embeddingError) {
            console.error("تعذر إنشاء embedding:", embeddingError);
          }
        }

        // Create a unique path for this user's file
        const extension = file.name.includes(".")
  ? file.name.split(".").pop().toLowerCase().replace(/[^a-z0-9]/g, "")
  : "bin";

const safeName = `${Date.now()}_${Math.random().toString(36).slice(2)}.${extension}`;
        const uniqueName =
          typeof crypto !== "undefined" && crypto.randomUUID
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

        storagePath = `${userId}/${uniqueName}_${safeName}`;

        // Upload original file to private Storage bucket
        const { error: storageError } = await supabase.storage
          .from("archive-files")
          .upload(storagePath, file, {
            contentType: file.type || "application/octet-stream",
            upsert: false,
          });

        if (storageError) throw storageError;

        // Save extracted text and metadata in Database
        const { data, error: databaseError } = await supabase
          .from("archive_files")
          .insert({
            user_id: userId,
            name: file.name,
            content: text || "",
            file_size: file.size,
            mime_type: file.type || "",
            storage_path: storagePath,
            embedding,
          })
          .select()
          .single();

        if (databaseError) throw databaseError;

        processedFiles.push({
          id: data.id,
          name: data.name,
          size: data.file_size || 0,
          type: data.mime_type || "",
          text: data.content || "",
          embedding: Array.isArray(data.embedding)
            ? data.embedding
            : null,
          storagePath: data.storage_path,
imageUrl: file.type.startsWith("image/")
  ? await createImagePreviewUrl(data.storage_path)
  : null,
pdfUrl: null,
error: "",
        });

        successCount++;
      } catch (error) {
        console.error(`تعذر حفظ الملف ${file.name}:`, error);

        // Remove uploaded Storage file if Database insertion failed
        if (storagePath) {
          await supabase.storage
            .from("archive-files")
            .remove([storagePath]);
        }

        processedFiles.push({
          id: `failed-${Date.now()}-${Math.random()}`,
          name: file.name,
          size: file.size,
          type: file.type || "",
          text: "",
          embedding: null,
          storagePath: null,
          pdfUrl: null,
          error: error.message || "تعذرت معالجة الملف أو حفظه.",
        });
      }
    }

    setFiles((previous) => [
      ...processedFiles.filter((file) => !file.error),
      ...processedFiles.filter((file) => file.error),
      ...previous,
    ]);

    setStatus(
      `انتهت العملية: تم حفظ ${successCount} من ${selectedFiles.length} ملف في حسابك.`
    );

    setLoading(false);
    event.target.value = "";
  };

  // Delete a file from Database and Storage
  const removeFile = async (id) => {
    const userId = session?.user?.id;
    const fileToRemove = files.find((file) => file.id === id);

    if (!userId || !fileToRemove) return;

    // Failed local uploads are not stored in Supabase
    if (String(id).startsWith("failed-")) {
      setFiles((previous) => previous.filter((file) => file.id !== id));
      if (expandedId === id) setExpandedId(null);
      return;
    }

    try {
      const { error: databaseError } = await supabase
        .from("archive_files")
        .delete()
        .eq("id", id)
        .eq("user_id", userId);

      if (databaseError) throw databaseError;

      if (fileToRemove.storagePath) {
        const { error: storageError } = await supabase.storage
          .from("archive-files")
          .remove([fileToRemove.storagePath]);

        if (storageError) {
          console.error("تعذر حذف الملف من Storage:", storageError);
        }
      }

      setFiles((previous) => previous.filter((file) => file.id !== id));

      if (expandedId === id) setExpandedId(null);

      setStatus("تم حذف الملف من أرشيفك.");
    } catch (error) {
      console.error("تعذر حذف الملف:", error);
      setStatus("تعذر حذف الملف: " + error.message);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} بايت`;

    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} كيلوبايت`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} ميجابايت`;
  };

  // Ask AI using the current user's archive
  const handleAskAI = async () => {
    if (!aiQuestion.trim() || aiLoading) return;

    const availableFiles = files
      .filter((file) => file.text?.trim() && !file.error)
      .map(({ name, text }) => ({ name, text }));

    if (availableFiles.length === 0) {
      setAiAnswer({
        answer: "ارفعي ملفًا يحتوي على نص أولًا.",
        sources: [],
      });
      return;
    }

    setAiLoading(true);
    setAiAnswer(null);

    try {
      setStatus("جارٍ تجهيز البحث الدلالي...");

      const queryEmbedding = await createEmbedding(aiQuestion.trim());
      const textChunks = createTextChunks(availableFiles);

      if (textChunks.length === 0) {
        throw new Error("لم يتم العثور على نصوص قابلة للبحث.");
      }

      const chunksWithEmbeddings = [];

      for (const chunk of textChunks) {
        const embedding = await createEmbedding(chunk.snippet);

        chunksWithEmbeddings.push({
          ...chunk,
          embedding,
        });
      }

      setStatus("جارٍ إرسال المقاطع إلى مساعدك الذكي...");

      const response = await fetch(
        "https://smart-archive-final.vercel.app/ask",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: aiQuestion.trim(),
            files: availableFiles,
            query_embedding: queryEmbedding,
            chunks: chunksWithEmbeddings,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "تعذّر الحصول على الإجابة."
        );
      }

      setAiAnswer(data);
      setStatus("اكتمل البحث الدلالي والإجابة عن السؤال.");
    } catch (error) {
      console.error("خطأ في المساعد الذكي:", error);

      setAiAnswer({
        answer: error.message || "تعذّر الاتصال بخادم الذكاء الاصطناعي.",
        sources: [],
      });

      setStatus("تعذرت معالجة السؤال.");
    } finally {
      setAiLoading(false);
    }
  };

  // Loading screen
  if (authLoading) {
    return (
      <div className="app" dir="rtl">
        <header className="hero">
          <div className="logo">✦</div>
          <h1>الأرشيف الذكي</h1>
          <p className="intro">جارٍ التحقق من حسابك...</p>
        </header>
      </div>
    );
  }

  // Login and signup screen
  if (!session) {
    return (
      <div className="app" dir="rtl">
        <header className="hero">
          <div className="logo">✦</div>
          <p className="eyebrow">مساحتك المعرفية</p>
          <h1>الأرشيف الذكي</h1>
          <p className="intro">
            نظّم ملفاتك، وابحث في محتوياتها، واسترجع معلوماتك بسهولة.
          </p>
        </header>

        <main className="auth-container">
          <h2>
            {authMode === "login" ? "مرحبًا بعودتك" : "إنشاء حساب جديد"}
          </h2>

          <p>
            {authMode === "login"
              ? "سجّل الدخول للوصول إلى أرشيفك الذكي."
              : "أنشئ حسابك وابدأ بتنظيم ملفاتك."}
          </p>

          <form onSubmit={handleAuth}>
            <input
              className="auth-input"
              type="email"
              placeholder="البريد الإلكتروني"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              required
            />

            <input
              className="auth-input"
              type="password"
              placeholder="كلمة المرور"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete={
                authMode === "login" ? "current-password" : "new-password"
              }
              minLength={6}
              required
            />

            <button
              className="auth-submit"
              type="submit"
              disabled={authBusy}
            >
              {authBusy
                ? "يرجى الانتظار..."
                : authMode === "login"
                  ? "تسجيل الدخول"
                  : "إنشاء الحساب"}
            </button>
          </form>

          {authMessage && (
            <p
              className={
                authMessage.includes("خطأ") ||
                authMessage.includes("يرجى") ||
                authMessage.includes("يجب")
                  ? "error-message"
                  : "status"
              }
              role="status"
            >
              {authMessage}
            </p>
          )}

          <button
            className="auth-switch"
            type="button"
            onClick={() => {
              setAuthMode(authMode === "login" ? "signup" : "login");
              setAuthMessage("");
            }}
          >
            {authMode === "login"
              ? "ليس لديك حساب؟ أنشئ حسابًا جديدًا"
              : "لديك حساب بالفعل؟ سجّل الدخول"}
          </button>
        </main>

        <footer>الأرشيف الذكي — تنظيم المعرفة في مكان واحد</footer>
      </div>
    );
  }

  // Wait until this user's cloud archive has loaded
  if (!filesLoaded) {
    return (
      <div className="app" dir="rtl">
        <header className="hero">
          <div className="logo">✦</div>
          <h1>الأرشيف الذكي</h1>
          <p className="intro">جارٍ تحميل أرشيفك...</p>
        </header>
      </div>
    );
  }

  // Main archive after login
  return (
    <div className="app" dir="rtl">
      <header className="hero">
        <div className="logo">✦</div>
        <p className="eyebrow">مساحتك المعرفية</p>
        <h1>الأرشيف الذكي</h1>
        <p className="intro">
          اجمع ملفاتك في مكان واحد، واستخرج النصوص منها، وابحث عن معلوماتك بسهولة.
        </p>

        <a className="primary-button" href="#archive">
          ابدأ الآن
        </a>

        <div style={{ marginTop: "18px" }}>
          <button
            className="primary-button"
            type="button"
            onClick={handleLogout}
          >
            تسجيل الخروج
          </button>
        </div>
      </header>

      <main className="content">
        <section className="stats">
          <article className="stat">
            <span>إجمالي الملفات</span>
            <strong>{files.length}</strong>
          </article>

          <article className="stat">
            <span>ملفات تمت معالجتها</span>
            <strong>{files.filter((file) => !file.error).length}</strong>
          </article>

          <article className="stat">
            <span>نتائج البحث</span>
            <strong>{filtered.length}</strong>
          </article>
        </section>

        <section className="panel" id="archive">
          <p className="eyebrow">إدارة المستندات</p>
          <h2>أضف ملفاتك</h2>

          <label className="upload-area">
            <span className="upload-icon">↑</span>
            <strong>اضغط هنا لاختيار الملفات</strong>
            <span>PDF أو DOCX أو TXT أو صور</span>

            <span className="button-label">
              {loading ? "جارٍ المعالجة..." : "اختيار الملفات"}
            </span>

            <input
              type="file"
              accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,image/*"
              multiple
              onChange={handleUpload}
              disabled={loading}
            />
          </label>

          {status && (
            <p className="status" role="status">
              {status}
            </p>
          )}
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">اكتشف معلوماتك</p>
              <h2>ابحث في الأرشيف</h2>
            </div>
            <span className="result-count">{filtered.length} ملف</span>
          </div>

          <input
            className="search-input"
            type="search"
            placeholder="اكتب كلمة أو عبارة للبحث..."
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            aria-label="البحث داخل الأرشيف"
          />

          {files.length === 0 ? (
            <div className="empty-state">
              <span>📚</span>
              <h3>أرشيفك يبدأ من هنا</h3>
              <p>ارفع أول ملف لتتمكن من استعراض نصوصه والبحث فيها.</p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="empty-state">
              <span>🔎</span>
              <h3>لم نجد نتائج</h3>
              <p>جرّب كلمة أخرى أو تأكد من محتوى الملفات.</p>
            </div>
          ) : (
            <div className="file-list">
              {filtered.map((file) => (
                <article className="file-card" key={file.id}>
                  <div className="file-header">
                    <div className="file-icon">📄</div>

                   <div className="file-details">
  <h3>{file.name}</h3>

  {file.type.startsWith("image/") && file.imageUrl && (
    <img
      src={file.imageUrl}
      alt={file.name}
      style={{
        display: "block",
        maxWidth: "100%",
        maxHeight: "260px",
        objectFit: "contain",
        margin: "12px 0",
        borderRadius: "10px",
      }}
    />
  )}
                      <p className="file-snippet">
                        {file.text
                          ? highlightText(
                              getSearchSnippet(file.text, query),
                              query
                            )
                          : "لا يوجد نص مستخرج من هذا الملف."}
                      </p>
                      <span>{formatSize(file.size)}</span>
                    </div>

                    <button
                      className="delete-button"
                      type="button"
                      onClick={() => removeFile(file.id)}
                    >
                      حذف
                    </button>
                  </div>

                  {file.error ? (
                    <p className="error-message">{file.error}</p>
                  ) : (
                    <>
                      <p className="text-preview">
                        {file.text
                          ? highlightText(
                              getSearchSnippet(file.text, query),
                              query
                            )
                          : "لم يتم العثور على نص داخل هذا الملف."}
                      </p>

                      <button
                        className="text-button"
                        type="button"
                        onClick={() =>
                          setExpandedId(
                            expandedId === file.id ? null : file.id
                          )
                        }
                      >
                        {expandedId === file.id
                          ? "إخفاء النص"
                          : "عرض النص المستخرج"}
                      </button>

                      {expandedId === file.id && (
                        <div className="extracted-text">
                          {file.text
                            ? highlightText(file.text, query)
                            : "لا يوجد نص مستخرج."}
                        </div>
                      )}
                    </>
                  )}
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="panel">
          <p className="eyebrow">مساعدك الذكي</p>
          <h2>اسأل أرشيفك</h2>
          <p>اسأل سؤالًا عن محتوى الملفات التي رفعتها.</p>

          <form
            onSubmit={(event) => {
              event.preventDefault();
              handleAskAI();
            }}
          >
            <input
              className="search-input"
              type="text"
              placeholder="مثال: ما أبرز طرق المحافظة على المياه؟"
              value={aiQuestion}
              onChange={(event) => setAiQuestion(event.target.value)}
              maxLength={2000}
              aria-label="سؤالك للذكاء الاصطناعي"
            />

            <button
              className="primary-button"
              type="submit"
              disabled={aiLoading}
            >
              {aiLoading ? "جارٍ التفكير..." : "اسأل الذكاء الاصطناعي"}
            </button>
          </form>

          {aiAnswer && (
            <div className="ai-answer" aria-live="polite">
              <h3>الإجابة</h3>
              <p>{aiAnswer.answer}</p>

              {aiAnswer.sources?.length > 0 && (
                <>
                  <h3>المصادر</h3>
                  {aiAnswer.sources.map((source, index) => (
                    <div key={`${source.fileName}-${index}`}>
                      <strong>{source.fileName}</strong>
                      <p>{source.snippet}</p>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </section>

        <section className="info-panel">
          <h2>عن الأرشيف الذكي</h2>
          <p>
            تتيح هذه النسخة رفع الملفات، واستخراج النصوص من المستندات والصور،
            والبحث عن الكلمات داخل محتوياتها.
          </p>
          <p>
            البحث الدلالي ومساعد RAG يستخدمان الآن Embeddings لفهم التشابه
            بين السؤال ومحتوى الملفات.
          </p>
        </section>
      </main>

      <footer>الأرشيف الذكي — تنظيم المعرفة في مكان واحد</footer>
    </div>
  );
}

export default App;
