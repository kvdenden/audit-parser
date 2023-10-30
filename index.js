require("dotenv").config();

const { OpenAI } = require("langchain/llms/openai");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { PDFLoader } = require("langchain/document_loaders/fs/pdf");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { RetrievalQAChain } = require("langchain/chains");

const llm = new OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const parsePDF = async (file) => {
  const loader = new PDFLoader(file, {
    splitPages: false,
  });

  const textSplitter = new RecursiveCharacterTextSplitter();
  const docs = await loader.loadAndSplit(textSplitter);

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());
  const chain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever());

  const query = "How many issues, grouped by severity, did the review find?";
  const response = await chain.call({
    query,
  });

  return response;
};

const file = process.argv[2];
parsePDF(file).then((res) => console.log(res));
