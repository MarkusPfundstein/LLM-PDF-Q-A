import * as fs from 'fs/promises';
import PDFParser from 'pdf-parse';

type PDFDocument = {
  text: string;
  pages: number;
  info: Record<string, any>;
  metadata: {
    version: string;
    contentLength: number;
  };
}

export async function extractPDFData(filePath: string): Promise<PDFDocument> {
  try {
    // Read the PDF file
    const dataBuffer = await fs.readFile(filePath);
    
    // Parse the PDF
    const data = await PDFParser(dataBuffer, {
      // Optional settings for pdf-parse
      pagerender: undefined, // Use default rendering
      max: 0, // 0 = unlimited pages
    });

    return {
      text: data.text,
      pages: data.numpages,
      info: data.info,
      metadata: {
        version: data.version,
        contentLength: data.contentLength
      }
    };
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`Failed to process PDF: ${error.message}`);
    }
    throw new Error('Failed to process PDF: Unknown error');
  }
} 