
import {
  createEmbedding,
  cosineSimilarity,
} from "./semanticSearch.js";

async function testSemanticSearch() {
  console.log("جارٍ تحميل نموذج البحث الدلالي...");

  const text1 = "أحفظ المستندات والملفات لاسترجاعها لاحقًا";
  const text2 = "كيف أجد ملفاتي التي خزنتها من قبل؟";

  const vector1 = await createEmbedding(text1);
  const vector2 = await createEmbedding(text2);

  const similarity = cosineSimilarity(vector1, vector2);

  console.log("عدد أبعاد المتجه:", vector1.length);
  console.log("درجة التشابه:", similarity);
  console.log("نجح اختبار المحرك!");
}

testSemanticSearch().catch(console.error);
