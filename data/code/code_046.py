export function longestPrefixLookup(input: string, trie: TrieNode): number[] {
  const result: number[] = [];
  let node = trie;
  for (const ch of input) {
    if (!node.children[ch]) break;
    node = node.children[ch];
    if (node.tokenId !== undefined) result.push(node.tokenId);
  }
  return result;
}
