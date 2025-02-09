import OpenAI from 'openai';

const CHUNK_ANALYZER_MODEL = "gpt-3.5-turbo";
const QUESTION_ANSWERING_MODEL = "o1-mini";

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY || '',
});

interface ChunkAnalysis {
    confidence: number;
    save_for_later_processing: boolean;
    summary: string;
    relevant_lines: number[];
    chunk_index: number;
}

export type GraphNode = {
    subject: string;
    predicate: string;
    value: string;
    chunk: string;
}

async function chooseNextGraphNode(nodes: GraphNode[], question: string, number_of_nodes: number = 10): Promise<GraphNode[]> {

    const prompt = `Our task is to answer the question: "${question}"

You are given a list of nodes, each of which is a triple of the form (subject, predicate, object).

Your task is to choose the next nodes to explore taken the question into account. Choose up to ${number_of_nodes} nodes. The
nodes you can choose are of the third entry in each triple. The second entry is the relation between the first and third entry. 

Its important to take into account the question when choosing the next nodes to explore. The less nodes we must explore before
finding the answer, the better. Return a json list containg the node names and the corresponding fourth entry.
If you don't find any nodes, return an empty array.

EXAMPLE_START
Nodes:
[
    { 
        "subject": "Summer", 
        "predicate: "takes place in",
        "value": "Colorado", 
        "chunk": "graph_0"
        },
    { 
        "subject": "Radio City", 
        "predicate: "located in",
        "value": "India", 
        "chunk": "graph_1"
    },
    { 
        "subject": "Radio City", 
        "predicate: "launched in",
        "value": "Summer 2001", 
        "chunk": "graph_2"
    }
]

Question: What is the date of Radio City's launch?

Output:
[
    {
        "subject": "Radio City",
        "predicate: "launched in",
        "value": "Summer 2001",
        "chunk": "graph_2"
    }
]


EXAMPLE_END

Return ONLY the JSON text, no other text such as \`\`\`json or whatever.

Nodes:
${JSON.stringify(nodes, null, 2)}

Question: ${question}

Output:
`
    //console.log(prompt);

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
                ],
                temperature: 0.0
            });

            const content = response.choices[0].message.content;
            
            const entities = JSON.parse(content || '[]');
            return entities;
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
    throw new Error('Failed to extract entities');
}

async function extractEntities(text: string): Promise<string[]> {
    const prompt = `Your task is to extract named entities from the given paragraph.
Respond with a JSON string[] of entities. Do not include any other text in your response and don't change
the type.

1-shot EXAMPLE:
Text: Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English
and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music
portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related
features.

Entities:
["Radio City", "India", "3 July 2001", "Hindi","English", "May 2008", "PlanetRadiocity.com"]
END EXAMPLE

END 1-shot EXAMPLE

Now, extract entities from this paragraph
Text: ${text}

Entities:
`

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
                ],
                temperature: 0.0
            });

            const content = response.choices[0].message.content;
            
            const entities = JSON.parse(content || '[]');
            return entities;
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
    throw new Error('Failed to extract entities');
}

async function extractRDFTriples(text: string, entities: string[]): Promise<[string, string, string][]> {
    const prompt = `Instruction:
Your task is to construct an RDF (Resource Description Framework) graph from the given passages and
named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.
Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each
passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
Convert the paragraph into a JSON list of triples of type string[]. Don't include any other text in your response.

1-shot EXAMPLE:
Text: Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English
and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music
portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related
features.

Entities: ["Radio City", "India", "3 July 2001", "Hindi","English", "May 2008", "PlanetRadiocity.com"]

Triples:
[
    ["Radio City", "located in", "India"],
    ["Radio City", "is", "private FM radio station"],
    ["Radio City", "started on", "3 July 2001"],
    ["Radio City", "plays songs in", "Hindi"],
    ["Radio City", "plays songs in", "English"],
    ["Radio City", "forayed into", "New Media"],
    ["Radio City", "launched", "PlanetRadiocity.com"],
    ["PlanetRadiocity.com", "launched in", "May 2008"],
    ["PlanetRadiocity.com", "is", "music portal"],
    ["PlanetRadiocity.com", "offers", "news"],
    ["PlanetRadiocity.com", "offers", "videos"],
    ["PlanetRadiocity.com", "offers", "songs"]
]
END 1-shot EXAMPLE

Now, create RDF triples for this text and entities:
Text: ${text}
Entities: ${JSON.stringify(entities)}
Triples
`

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
                ],
                temperature: 0.0
            });

            const content = response.choices[0].message.content;
            
            const rdfData = JSON.parse(content || '{"named_entities": [], "triples": []}');
            return rdfData;
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
    throw new Error('Failed to extract RDF triples');
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
    const prompt = `You are a critical reader. Given this text chunk from a PDF:
      
"${chunk}"

And this question: "${question}"

your task is to judge whether this chunk contains information relevant to answering the question. Be very critical and only return true if you are very confident that the chunk contains relevant information

Return your analysis as a JSON object with these fields:
- summary (string): Explain why you think this chunk is relevant or not to the question in 1 or 2 sentences maximum.
- confidence (number between 0-1): How confident are you that this chunk contains relevant information
- save_for_later_processing (boolean): Should we use this chunk to form the final answer?
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

export type QAInput = {
    chunk: string;
    index: number;
}

async function llm_do_question_answer(qaInput: QAInput[], question: string, max_characters: number): Promise<string> {
    // Build context and track chunk indices
    let context = '';
    let chunkIndices: number[] = [];
    for (const analysis of qaInput) {
        const chunk = analysis.chunk;
        if (context.length + chunk.length <= max_characters) {
            context += `[Chunk ${analysis.index}]:\n${chunk}\n\n`;
            chunkIndices.push(analysis.index);
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

export { extractEntities, extractRDFTriples, chooseNextGraphNode, llm_do_question_answer };