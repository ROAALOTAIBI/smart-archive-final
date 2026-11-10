
import "./App.css";
import { createEmbedding, cosineSimilarity } from "./semanticSearch.js";
import { useEffect, useMemo, useState } from "react";

import * as pdfjsLib from "pdfjs-dist";
import pdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import mammoth from "mammoth";
import { createWorker } from "tesseract.js";


pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

async function extractText(file) {

  const extension = file.name.split(".").pop().toLowerCase();

  if (extension === "txt") {
    return await file.text();
  }

  if (extension === "docx") {
    const buffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer: buffer });
    return result.value;
  }

  if (extension === "pdf") {
    const buffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
    const pages = [];
    const worker = await createWorker("ara+eng");

    try {
      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
        const page = await pdf.getPage(pageNumber);
        const viewport = page.getViewport({ scale: 5 });
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");

        canvas.width = Math.ceil(viewport.width);
        canvas.height = Math.ceil(viewport.height);

        await page.render({ canvasContext: context, viewport }).promise;
        const result = await worker.recognize(canvas);
        pages.push(result.data.text);
        canvas.width = 0;
        canvas.height = 0;
      }

      return pages.join("\n\n");
    } finally {
      await worker.terminate();
    }
  }

  if (file.type.startsWith("image/")) {
    const worker = await createWorker("ara+eng");

    try {
      const image = await createImageBitmap(file);
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
     const scale = 3;

      canvas.width = image.width * scale;
      canvas.height = image.height * scale;
      context.drawImage(image, 0, 0, canvas.width, canvas.height);

      const result = await worker.recognize(canvas);
      return result.data.text;
    } finally {
      await worker.terminate();
    }
  }

  throw new Error("نوع الملف غير مدعوم.");
}

function highlightText(text, query) {
  const words = [...new Set(query.trim().split(/\s+/).filter(Boolean))];

  if (words.length === 0) {
    return text;
  }

  const escapedWords = words.map((word) =>
    word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  );

  const regex = new RegExp(`(${escapedWords.join("|")})`, "gi");

  return text.split(regex).map((part, index) =>
    words.some((word) => part.toLowerCase() === word.toLowerCase()) ? (
      <mark key={index}>
        {part}
      </mark>
    ) : (
      part
    )
  );
}


function getSearchSnippet(text, query, maxLength = 180) {
  const source = (text || "").replace(/\s+/g, " ").trim();
  const words = (query || "").trim().split(/\s+/).filter(Boolean);

  if (!source) return "";

  if (words.length === 0) {
    return source.slice(0, maxLength) +
      (source.length > maxLength ? "..." : "");
  }

  const lowerText = source.toLowerCase();
  let firstIndex = -1;

  for (const word of words) {
    const index = lowerText.indexOf(word.toLowerCase());

    if (index !== -1 && (firstIndex === -1 || index < firstIndex)) {
      firstIndex = index;
    }
  }




  if (firstIndex === -1) {
    return source.slice(0, maxLength) +
      (source.length > maxLength ? "..." : "");
  }

  const start = Math.max(0, firstIndex - 60);
  const end = Math.min(source.length, start + maxLength);

  return (
    (start > 0 ? "..." : "") +
    source.slice(start, end) +
    (end < source.length ? "..." : "")
  );
}
function App() {
  const [files, setFiles] = useState(() => {
  try {
    return JSON.parse(localStorage.getItem("smart-archive-files") || "[]");
  } catch {
    return [];
  }
});
  const [query, setQuery] = useState("");
  const [aiQuestion, setAiQuestion] = useState("");
const [aiAnswer, setAiAnswer] = useState(null);
const [aiLoading, setAiLoading] = useState(false);
  const [semanticResults, setSemanticResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
  try {
    localStorage.setItem("smart-archive-files", JSON.stringify(files));
  } catch (error) {
    console.error("تعذر حفظ الأرشيف:", error);
  }
}, [files]);
  


const lexicalFiltered = useMemo(() => {
  const searchTerm = query.trim().toLowerCase();

  if (!searchTerm) {
    return files;
  }

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

const filtered = query.trim()
  ? (semanticResults ?? lexicalFiltered).filter((file) => file.score > 0)
  : files;
console.log("Semantic results:", semanticResults);


  const handleUpload = async (event) => {
    const selectedFiles = Array.from(event.target.files || []);

    if (selectedFiles.length === 0) {
      return;
    }

    setLoading(true);
    setStatus("جارٍ استخراج النصوص من الملفات...");

    const processedFiles = [];

    for (const file of selectedFiles) {
      const id = `${Date.now()}-${Math.random()}`;

      try {
       
const text = await extractText(file);

let embedding = null;

if (text && text.trim()) {
  try {
    embedding = await createEmbedding(text.slice(0, 4000));
  } catch (embeddingError) {
    console.error("تعذر إنشاء embedding:", embeddingError);
  }
}

processedFiles.push({
  id,
  name: file.name,
  size: file.size,
  text: text || "",
  embedding,
  error: "",
});

      } catch (error) {
        processedFiles.push({
          id,
          name: file.name,
          size: file.size,
          text: "",
          error: error.message || "تعذرت معالجة الملف.",
        });
      }
    }

    setFiles((previous) => [...processedFiles, ...previous]);

    const successCount = processedFiles.filter(
      (file) => !file.error
    ).length;

    setStatus(
      `انتهت المعالجة: نجح ${successCount} من ${processedFiles.length} ملف.`
    );

    setLoading(false);
    event.target.value = "";
  };

  const removeFile = (id) => {
    setFiles((previous) => previous.filter((file) => file.id !== id));

    if (expandedId === id) {
      setExpandedId(null);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) {
      return `${bytes} بايت`;
    }

    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} كيلوبايت`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} ميجابايت`;
  };

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
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: aiQuestion.trim(),
        files: availableFiles,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "تعذّر الحصول على الإجابة."
      );
    }

    setAiAnswer(data);
  } catch (error) {
    setAiAnswer({
      answer:
        error.message || "تعذّر الاتصال بخادم الذكاء الاصطناعي.",
      sources: [],
    });
  } finally {
    setAiLoading(false);
  }
};

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
<p className="file-snippet">
  {file.text
    ? highlightText(getSearchSnippet(file.text, query), query)
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
                          {file.text || "لا يوجد نص مستخرج."}
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
<form onSubmit={(event) => {
  event.preventDefault();
  
  handleAskAI();
}}>
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
            البحث الحالي يعتمد على الكلمات؛ البحث الدلالي وRAG يحتاجان إلى تطوير إضافي.
          </p>
        </section>
      </main>

      <footer>الأرشيف الذكي — تنظيم المعرفة في مكان واحد</footer>
    </div>
  );
}

export default App;
