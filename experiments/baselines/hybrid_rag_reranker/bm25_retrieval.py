    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List[Document], List[float]]:
        """Retrieve top-k documents for a query using BM25.

        Args:
            query: Query string
            top_k: Number of documents to return

        Returns:
            Tuple of (documents, scores) sorted by score descending, with non-positive scores filtered out
        """
        scores = self.get_scores(query)

        # Get indices of top-k scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Get documents and scores for top indices, filtering out non-positive scores
        filtered_docs = []
        filtered_scores = []
        for idx in top_indices:
            if scores[idx] > 0:
                filtered_docs.append(self.documents[idx])
                filtered_scores.append(scores[idx])

        return filtered_docs, filtered_scores