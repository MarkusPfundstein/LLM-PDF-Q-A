import { config } from 'dotenv';
config(); // Load environment variables

import { extractPDFData } from './pdf';
import OpenAI from 'openai';
import N3 from 'n3';

const { DataFactory } = N3;
const { namedNode, literal, quad } = DataFactory;

import { VectorStore } from './vector';
import fs from 'fs/promises';
import { chooseNextGraphNode, extractEntities, extractRDFTriples, GraphNode, llm_do_question_answer, QAInput } from './prompts';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY || '',
});

const rdfStore = new N3.Store();
const vectorStore = new VectorStore();

const EMBEDDING_MODEL = "text-embedding-ada-002"

const VECTOR_STORE_PATH = '.vector.json';
const RDF_STORE_PATH = '.db.rdf';

function chunkPDF(text: string, max_chunk_length: number = 500): string[] {
    // Split by multiple newlines and filter out empty chunks
    const initialChunks = text
        .split(/\n{2,}/)
        .map(chunk => chunk.trim())
        .filter(chunk => chunk.length > 0);

    // Further split chunks that exceed max length
    const finalChunks: string[] = [];
    for (const chunk of initialChunks) {
        if (chunk.length <= max_chunk_length) {
            finalChunks.push(chunk);
        } else {
            // Split long chunks into smaller pieces at sentence boundaries if possible
            const sentences = chunk.match(/[^.!?]+[.!?]+/g) || [chunk];
            let currentChunk = '';
            
            for (const sentence of sentences) {
                if ((currentChunk + sentence).length <= max_chunk_length) {
                    currentChunk += sentence;
                } else {
                    if (currentChunk) {
                        finalChunks.push(currentChunk.trim());
                    }
                    // If a single sentence is longer than max_chunk_length, split it by words
                    if (sentence.length > max_chunk_length) {
                        const words = sentence.split(' ');
                        currentChunk = '';
                        for (const word of words) {
                            if ((currentChunk + ' ' + word).length <= max_chunk_length) {
                                currentChunk += (currentChunk ? ' ' : '') + word;
                            } else {
                                if (currentChunk) {
                                    finalChunks.push(currentChunk.trim());
                                }
                                currentChunk = word;
                            }
                        }
                    } else {
                        currentChunk = sentence;
                    }
                }
            }
            if (currentChunk) {
                finalChunks.push(currentChunk.trim());
            }
        }
    }

    return finalChunks;
}

function extractChunkIndices(text: string): number[] {
    // Create a Set to store unique chunk indices
    const indices = new Set<number>();
    
    // Regular expression to match chunk indices pattern
    const pattern = /\{chunk_indices:\s*\[([\d,\s]+)\]\}/g;
    
    // Find all matches in the text
    const matches = text.matchAll(pattern);
    
    // Process each match
    for (const match of matches) {
        // Extract the numbers from the array string and convert to numbers
        const numbers = match[1]
            .split(',')
            .map(num => parseInt(num.trim()))
            .filter(num => !isNaN(num));
            
        // Add numbers to the Set
        numbers.forEach(num => indices.add(num));
    }
    
    // Convert Set to sorted array and return
    return Array.from(indices)//.sort((a, b) => a - b);
}

async function getEmbedding(text: string): Promise<number[]> {
    try {
        const response = await openai.embeddings.create({
            model: EMBEDDING_MODEL,
            input: text,
            encoding_format: "float"
        });

        // Return the embedding vector
        return response.data[0].embedding;
    } catch (error) {
        console.error('Error getting embedding:', error);
        throw new Error(`Failed to get embedding: ${error}`);
    }
}

async function indexPDF(pdfPath: string, max_chunks: number = -1): Promise<boolean> {
    const pdfData = await extractPDFData(pdfPath);
    
    const chunks = chunkPDF(pdfData.text);

    const { DataFactory } = N3;
    const { namedNode, literal, quad } = DataFactory;
    
    // Extract entities and RDF triples from each chunk
    max_chunks = max_chunks == -1 ? chunks.length : max_chunks;
    for (let i = 0; i < max_chunks; i++) {
        try {
            const chunk = chunks[i];
            console.log(`Processing chunk ${i + 1}/${max_chunks} (of ${chunks.length})`);
            
            const chunkEntities = await extractEntities(chunk);

            for (const entity of chunkEntities) {
                const embedding = await getEmbedding(entity);

                vectorStore.store(entity, embedding);

                // need to go through all entities in the vector store and find synonyms. 
                // we need to do this because the synonyms might not be in the same chunk
                // and we need to find them all and add them to the RDF store
                const similarities = vectorStore.get_most_similar(entity, embedding, 1);

                // Only consider it a synonym if similarity is above a threshold
                // Maybe use a LLM here to judge?
                const SYNONYM_THRESHOLD = 0.95;
                if (similarities.length > 0 && similarities[0].similarity >= SYNONYM_THRESHOLD) {
                    const synQuad = quad(
                        namedNode(entity), // Subject
                        namedNode('is_synonym'), // Predicate
                        literal(similarities[0].text, 'en'), // Object
                        namedNode(`chunk_${i}`), // Graph
                    );
                    rdfStore.addQuad(synQuad);
                }
            }
            
            // extract RDF triples from chunk using entities
            const triples = await extractRDFTriples(chunk, chunkEntities);

            // Add triples to RDF store
            for (const triple of triples) {
                const relQuad = quad(
                    namedNode(triple[0]), // Subject
                    namedNode(triple[1]), // Predicate
                    literal(triple[2], 'en'), // Object
                    namedNode(`chunk_${i}`), // Graph
                );
                //console.log(`\nQuad:`, myQuad);
                rdfStore.addQuad(relQuad);
            }
        } catch (error) {
            console.error('Error processing chunk:', i, error);
        }
    }

    return true;
}

async function queryPDF(pdfPath: string, question: string, max_similarities: number = 3): Promise<void> {
    let depth = 0;

    const strategy : 'embed_question' | 'embed_entities' = 'embed_question';

    let allNodes: GraphNode[] = [];
    if (strategy == 'embed_question') {
        const embedding = await getEmbedding(question);
        const similarities = vectorStore.get_most_similar(question, embedding, 10);

        for (const similarity of similarities) {
            console.log(`\nSimilarity:`, similarity.text);
            const nodes = rdfStore.match(namedNode(similarity.text), null, null);
            
            for (const node of nodes) {
                console.log(node.subject.value, node.predicate.value, node.object.value);
                allNodes.push({
                    "subject": node.subject.value,
                    
                    "predicate": node.predicate.value,
                    "value": node.object.value,
                    "chunk": node.graph.value
                });
            }
        }
    } else if (strategy == 'embed_entities') {
        const entities = await extractEntities(question);
        console.log('\nEntities:', entities);
        for (const entity of entities) {

            let embedding = vectorStore.getEmbedding(entity);
            if (!embedding) {
                embedding = await getEmbedding(entity);
                vectorStore.store(entity, embedding);
            }
            const similarities = vectorStore.get_most_similar(question, embedding, max_similarities);

            for (const similarity of similarities) {
                console.log(`\nSimilarity:`, similarity.text);
                const nodes = rdfStore.match(namedNode(similarity.text), null, null);
                
                for (const node of nodes) {
                    console.log(node.subject.value, node.predicate.value, node.object.value);
                    allNodes.push({
                        "subject": node.subject.value,

                        "predicate": node.predicate.value,
                        "value": node.object.value,
                        "chunk": node.graph.value
                    });
                }
            }
        }
    }

    const visitedNodes: GraphNode[] = [];
    // search 3 levels deep
    while (true) {
        if (depth > 2) {
            console.log('break because 3 levels reached');
            break;
        }
        if (allNodes.length == 0) {
            console.log('break because no nodes left');
            break;
        }

        const nextNodes = await chooseNextGraphNode(allNodes, question, allNodes.length);
        console.log('\nNext Nodes:\n', nextNodes.map(n => `${n.subject} ${n.predicate} ${n.value}\n`));
        allNodes = [];
        for (const node of nextNodes) {
            console.log('\Follow node:', node);
            const queryNodes1 = rdfStore.match(null, null, literal(node.value, "en"));
            const queryNodes2 = rdfStore.match(namedNode(node.subject), null, null);
            const queryNodes = [...queryNodes1, ...queryNodes2];
            console.log('\Try visit: ', queryNodes.length);
            const tmp: GraphNode[] = [];
            for (const n of queryNodes) {
                if (!visitedNodes.some(v => {
                    return (v.subject == n.subject.value 
                            && v.predicate == n.predicate.value
                            && v.value == n.object.value
                            && v.chunk == n.graph.value);
                })) {
                    console.log('New node:', n.subject.value, n.predicate.value, n.object.value, n.graph.value);
                    tmp.push({
                        "subject": n.subject.value,
                        "predicate": n.predicate.value,
                        "value": n.object.value,
                        "chunk": n.graph.value
                    });
                    visitedNodes.push(tmp[tmp.length - 1]);
                }
            }
            allNodes.push(...tmp);
        }
        //console.log('\nIteration', depth + 1, 'All Nodes:', allNodes);
        console.log(`iteration: ${depth + 1} visitedNodes: ${visitedNodes.length}`);        
        depth++;
    }

    const chunkIndices = visitedNodes.map(n => n.chunk.split('_')[1]);
    const pdfData = await extractPDFData(pdfPath);
    const chunks = chunkPDF(pdfData.text);
    const qaInput: QAInput[] = chunkIndices.map(index => {
        return {
            chunk: chunks[parseInt(index)],
            index: parseInt(index)
        }
    });

    // Add the question answering step
    const answer = await llm_do_question_answer(qaInput, question, 20000);
    console.log('Answer:', answer);
    const extractedChunkIndices = extractChunkIndices(answer);
    console.log('\nExtracted Chunk Indices:', extractedChunkIndices);    
}

async function saveRDFStore(pdfPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        const writer = new N3.Writer({ format: 'N-Triples' });
        for (const match of rdfStore.match(null, null, null)) {
            writer.addQuad(match);
        }
        writer.end(async (error, result) => {
            if (error) {
                reject(error);
            } else {
                await fs.writeFile(pdfPath + '.rdf', result);
                resolve(result);
            }
        });
    });
}

async function loadRDFStoreFromText(rdfText: string): Promise<void> {    
    // Split the text into lines and filter out empty lines
    const lines = rdfText.split('\n').filter(line => line.trim().length > 0);
    
    for (const line of lines) {
        // Parse each line in format: <subject> <predicate> "object"@en <graph> .
        const match = line.match(/<([^>]+)>\s+<([^>]+)>\s+"([^"]+)"@en\s+<([^>]+)>\s+\./);
        
        if (match) {
            const [_, subject, predicate, object, graph] = match;
            
            const rdfQuad = quad(
                namedNode(subject),
                namedNode(predicate),
                literal(object, 'en'),
                namedNode(graph)
            );
            
            rdfStore.addQuad(rdfQuad);
        }
    }
}

async function loadRDFStore(pdfPath: string): Promise<void> {
    try {
        const rdfFile = await fs.readFile(pdfPath + '.rdf', 'utf-8');
        await loadRDFStoreFromText(rdfFile);
    } catch (error: any) {
        if (error.code === 'ENOENT') {
            console.log(`No existing RDF store found at ${pdfPath}.rdf`);
            return;
        }
        throw new Error(`Failed to load RDF store: ${error.message}`);
    }
}

async function main(): Promise<void> {
    try {
        const args = process.argv.slice(2);
        const pdfPath = args[0];

        if (!pdfPath) {
            console.error('Please provide a PDF file path as argument');
            process.exit(1);
        }

        // Check for indexing mode
        if (args.includes('-i')) {
            console.log('Indexing PDF:', pdfPath);
            await indexPDF(pdfPath);
            console.log('Indexing complete');
            try {
                await vectorStore.save(VECTOR_STORE_PATH);
                await saveRDFStore(RDF_STORE_PATH);
            } catch (error) {
                console.error('Error saving vector store:', error);
            }        
            process.exit(0);
        }
        
        // Check for querying mode
        const questionIndex = args.indexOf('-q');
        if (questionIndex !== -1 && args[questionIndex + 1]) {
            await vectorStore.load(VECTOR_STORE_PATH);
            await loadRDFStore(RDF_STORE_PATH);
            await queryPDF(pdfPath, args[questionIndex + 1]);
            process.exit(0);
        }

        console.error('Please specify either -i to index or -q "question" to query');
        process.exit(1);
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

main(); 