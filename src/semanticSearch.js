
import { pipeline } from "@huggingface/transformers";

let extractorPromise = null;

async function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = pipeline(
      "feature-extraction",
      "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
    );
  }

  return extractorPromise;
}

export async function createEmbedding(text) {
  const extractor = await getExtractor();

  const output = await extractor(text, {
    pooling: "mean",
    normalize: true,
  });

  return Array.from(output.data);
}

export function cosineSimilarity(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) {
    throw new Error("لا يمكن مقارنة متجهين بأبعاد مختلفة.");
  }

  let score = 0;

  for (let i = 0; i < vectorA.length; i++) {
    score += vectorA[i] * vectorB[i];
  }

  return score;
}
