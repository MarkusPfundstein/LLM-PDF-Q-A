declare module 'pdf-parse' {
  interface PDFData {
    text: string;
    numpages: number;
    info: Record<string, any>;
    metadata: Record<string, any>;
    version: string;
    contentLength: number;
  }

  function PDFParse(dataBuffer: Buffer, options?: {
    pagerender?: Function;
    max?: number;
  }): Promise<PDFData>;

  export = PDFParse;
} 