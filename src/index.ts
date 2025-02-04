import { config } from 'dotenv';
config(); // Load environment variables

import { extractPDFData } from './pdf';
import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY || '',
});

const CHUNK_ANALYZER_MODEL = "gpt-3.5-turbo";
const QUESTION_ANSWERING_MODEL = "o1-mini";

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

interface ChunkAnalysis {
    confidence: number;
    save_for_later_processing: boolean;
    summary: string;
    relevant_lines: number[];
    chunk_index: number;
}

async function llm_do_initial_pass(chunks: string[], question: string, batchSize: number = 3): Promise<ChunkAnalysis[]> {
    try {
        const analyses: ChunkAnalysis[] = [];
        
        // Process chunks in batches
        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            const batchPromises = batch.map((chunk, batchIndex) => {
                const chunkIndex = i + batchIndex;
                return analyzeChunk(chunk, question, chunkIndex);
            });
            
            // Wait for all promises in the current batch to resolve
            const batchResults = await Promise.all(batchPromises);
            analyses.push(...batchResults);
        }

        return analyses;
    } catch (error) {
        throw new Error(`OpenAI API error: ${error}`);
    }
}

// Helper function to analyze a single chunk
async function analyzeChunk(chunk: string, question: string, chunkIndex: number): Promise<ChunkAnalysis> {
    const prompt = `Given this text chunk from a PDF:
      
"${chunk}"

And this question: "${question}"

Please analyze whether this chunk contains information relevant to answering the question.
Return your analysis as a JSON object with these fields:
- confidence (number between 0-1): How confident are you that this chunk contains relevant information
- save_for_later_processing (boolean): Should we use this chunk to form the final answer?
- summary (string): Summarize this chunk in 1-2 sentences maximum
- relevant_lines (array of numbers): Line numbers in this chunk that contain relevant information

Return ONLY the JSON text, no other text such as \`\`\`json or whatever.`;

    let attempts = 0;
    while (attempts < 3) {
        try {
            const response = await openai.chat.completions.create({
                model: CHUNK_ANALYZER_MODEL,
                messages: [
                    {
                        role: "user",
                        content: prompt
                    }
                ]
            });

            console.log(response.choices[0].message.content);
            const analysis = JSON.parse(response.choices[0].message.content || '{}');
            analysis.chunk_index = chunkIndex;
            return analysis;
        } catch (error) {
            if (error instanceof SyntaxError) {
                console.error('JSON parsing error:', error);
                attempts++;
                if (attempts === 3) {
                    throw new Error('Failed to parse JSON response after 3 attempts');
                }
            } else {
                throw error;
            }
        }
    }
    throw new Error('Failed to analyze chunk');
}

async function llm_do_question_answer(filteredAnalysis: ChunkAnalysis[], chunks: string[], question: string, max_characters: number): Promise<string> {
    // Build context and track chunk indices
    let context = '';
    let chunkIndices: number[] = [];
    for (const analysis of filteredAnalysis) {
        const chunk = chunks[analysis.chunk_index];
        if (context.length + chunk.length <= max_characters) {
            context += `[Chunk ${analysis.chunk_index}]:\n${chunk}\n\n`;
            chunkIndices.push(analysis.chunk_index);
        } else {
            break;
        }
    }

    const prompt = `Based on these relevant excerpts from a document:

${context}

Please answer this question: "${question}"

For each sentence in your answer, add a list of chunk indices that were used to generate that sentence. Use this format:
[Sentence] {chunk_indices: [x,y,z]}

For example:
"The cat is black. {chunk_indices: [0,2]} It likes to play with yarn. {chunk_indices: [1]}"

Provide a clear and concise answer based only on the information given in the excerpts.
The available chunk indices are: ${JSON.stringify(chunkIndices)}`;

    try {
        const response = await openai.chat.completions.create({
            model: QUESTION_ANSWERING_MODEL,
            messages: [{ role: "user", content: prompt }]
        });

        return response.choices[0].message.content || 'No answer generated';
    } catch (error) {
        throw new Error(`OpenAI API error in question answering: ${error}`);
    }
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

async function main(): Promise<void> {
    try {
        const args = process.argv.slice(2);
        const pdfPath = args[0];
        let question: string = "";

        const questionIndex = args.indexOf('-q');
        if (questionIndex !== -1 && args[questionIndex + 1]) {
            question = args[questionIndex + 1];
        }
        if (!question) {
            console.error('Please provide a question as argument');
            process.exit(1);
        }

        if (!pdfPath) {
            console.error('Please provide a PDF file path as argument');
            process.exit(1);
        }

        
        const pdfData = await extractPDFData(pdfPath);
        console.log('PDF Processing Results:');
        console.log(pdfData);
        
        const chunks = chunkPDF(pdfData.text);
        
        // Process 5 chunks in parallel
        const analysis = await llm_do_initial_pass(chunks, question, 5);
        const filteredAnalysis = analysis
            .filter(chunk => chunk.save_for_later_processing)
            .sort((a, b) => b.confidence - a.confidence);
        console.log('\nGPT Analysis:', JSON.stringify(filteredAnalysis, null, 2));

        // Add the question answering step
        const answer = await llm_do_question_answer(filteredAnalysis, chunks, question, 5000);
        console.log('\nAnswer:', answer);

        const extractedChunkIndices = extractChunkIndices(answer);
        console.log('\nExtracted Chunk Indices:', extractedChunkIndices);

        // Get the referenced chunks from the original text
        const referencedChunks = extractedChunkIndices.map(index => chunks[index]);
        console.log('\nReferenced Chunks:');
        referencedChunks.forEach((chunk, i) => {
            console.log(`\nChunk ${extractedChunkIndices[i]}:`);
            console.log(chunk);
        });

        process.exit(0);
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

main(); 