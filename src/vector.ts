import fs from 'fs/promises';

interface VectorEntry {
    text: string;
    embedding: number[];
}

// Calculate cosine similarity with all entries
export type Similarity = {
    text: string;
    similarity: number;
};

export class VectorStore {
    private entries: VectorEntry[] = [];

    /**
     * Stores a text and its corresponding embedding vector
     */
    store(text: string, embedding: number[]): void {
        if (this.contains(text)) {
            return;
        }
        this.entries.push({ text, embedding });
    }

    /**
     * Checks if a text exists in the store
     */
    contains(text: string): boolean {
        return this.entries.some(entry => entry.text === text);
    }

    /**
     * Returns the most similar texts and their similarity scores
     * based on cosine similarity of embeddings
     */
    get_most_similar(text: string, embedding: number[], max_results: number = 10): Similarity[] {
        if (this.entries.length === 0) {
            return []
        }

        const similarities: Similarity[] = []
        for (const entry of this.entries) {
            if (entry.text === text) {
                continue
            }

            const cos = this.cosineSimilarity(entry.embedding, embedding)

            similarities.push({
                text: entry.text,
                similarity: cos
            })
        }

        // Sort by similarity in descending order
        similarities.sort((a, b) => b.similarity - a.similarity);

        return similarities.slice(0, max_results);
    }

    /**
     * Calculates the cosine similarity between two vectors
     */
    cosineSimilarity(vec1: number[], vec2: number[]): number {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have the same length');
        }

        const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
        const magnitude1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
        const magnitude2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));

        if (magnitude1 === 0 || magnitude2 === 0) {
            return 0;
        }

        return dotProduct / (magnitude1 * magnitude2);
    }

    /**
     * Calculates similarity between two texts that are stored in the vector store
     * @returns number between 0 and 1, where 1 means identical and 0 means no similarity
     * @throws Error if either text is not found in the store
     */
    calcSimilarity(text1: string, text2: string): number {
        const entry1 = this.entries.find(entry => entry.text === text1);
        const entry2 = this.entries.find(entry => entry.text === text2);

        if (!entry1) {
            throw new Error(`Text "${text1}" not found in store`);
        }
        if (!entry2) {
            throw new Error(`Text "${text2}" not found in store`);
        }

        return this.cosineSimilarity(entry1.embedding, entry2.embedding);
    }

    /**
     * Saves the vector store to a JSON file
     * @param filepath Path where to save the vector store
     */
    async save(filepath: string): Promise<void> {
        try {
            await fs.writeFile(
                filepath, 
                JSON.stringify({ entries: this.entries }, null, 2),
                'utf-8'
            );
            console.log(`Vector store saved to ${filepath}`);
        } catch (error: any) {
            throw new Error(`Failed to save vector store: ${error.message}`);
        }
    }

    /**
     * Loads the vector store from a JSON file
     * @param filepath Path to the vector store file
     */
    async load(filepath: string): Promise<void> {
        try {
            const data = await fs.readFile(filepath, 'utf-8');
            const parsed = JSON.parse(data);
            
            if (!parsed.entries || !Array.isArray(parsed.entries)) {
                throw new Error('Invalid vector store file format');
            }

            // Validate entries
            for (const entry of parsed.entries) {
                if (!entry.text || !Array.isArray(entry.embedding)) {
                    throw new Error('Invalid entry format in vector store file');
                }
            }

            this.entries = parsed.entries;
            console.log(`Vector store loaded from ${filepath}`);
        } catch (error: any) {
            if (error.code === 'ENOENT') {
                console.log(`No existing vector store found at ${filepath}`);
                this.entries = [];
            } else {
                throw new Error(`Failed to load vector store: ${error.message}`);
            }
        }
    }

    /**
     * Returns the embedding for a given text, or null if not found
     * @param text The text to look up
     * @returns number[] | null The embedding vector if found, null otherwise
     */
    getEmbedding(text: string): number[] | null {
        const entry = this.entries.find(entry => entry.text === text);
        return entry ? entry.embedding : null;
    }
}
