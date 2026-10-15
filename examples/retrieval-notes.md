# Retrieval notes

This demo stores embeddings in Chroma. Embeddings are stored on disk for similarity search.

The query API returns citations for the chunks it retrieved. Clients can also ask for a streamed answer, which arrives as server-sent events.

The default language-model provider is offline. It quotes the best matching source and does not call an external API.

Uploads accept PDF, Markdown, and plain text. Each file is split into overlapping chunks before those chunks are embedded.
