import java.util.*;

public class KruskalMST {
    public static MSTResult kruskal(int n, List<int[]> rawEdges) {
        if (n <= 0) {
            throw new IllegalArgumentException("Number of vertices must be positive");
        }
        Objects.requireNonNull(rawEdges, "Edges list must not be null");

        List<KruskalEdge> edges = new ArrayList<>();
        for (int[] e : rawEdges) {
            if (e == null || e.length != 3) {
                throw new IllegalArgumentException("Each edge must be an array of length 3: u, v, w");
            }
            int u = e[0], v = e[1], w = e[2];
            if (u < 0 || u >= n || v < 0 || v >= n) {
                throw new IllegalArgumentException("Vertex index out of range: " + Arrays.toString(e));
            }
            edges.add(new KruskalEdge(u, v, w));
        }
        edges.sort(Comparator.comparingInt(edge -> edge.w));

        DisjointSetUnion dsu = new DisjointSetUnion(n);
        List<KruskalEdge> mstEdges = new ArrayList<>();
        int totalWeight = 0;

        for (KruskalEdge edge : edges) {
            if (dsu.union(edge.u, edge.v)) {
                mstEdges.add(edge);
                totalWeight += edge.w;
                if (mstEdges.size() == n - 1) {
                    break;
                }
            }
        }

        if (mstEdges.size() != n - 1) {
            throw new IllegalStateException("Graph not connected");
        }

        return new MSTResult(totalWeight, mstEdges);
    }

    public static void main(String[] args) {
        int n = 4;
        List<int[]> edges = new ArrayList<>();
        edges.add(new int[] { 0, 1, 2 });
        edges.add(new int[] { 0, 2, 1 });
        edges.add(new int[] { 0, 3, 3 });
        edges.add(new int[] { 2, 3, 4 });
        edges.add(new int[] { 1, 3, 5 });

        MSTResult res = kruskal(n, edges);
        System.out.println("MST total weight = " + res.totalWeight);
        System.out.println("MST edges:");
        for (KruskalEdge edge : res.edges) {
            System.out.println(edge.u + " - " + edge.v + " (w=" + edge.w + ")");
        }
    }
}

class KruskalEdge {
    final int u;
    final int v;
    final int w;

    KruskalEdge(int u, int v, int w) {
        this.u = u;
        this.v = v;
        this.w = w;
    }
}

class DisjointSetUnion {
    private final int[] parent;
    private final int[] rank;

    DisjointSetUnion(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    boolean union(int a, int b) {
        int pa = find(a);
        int pb = find(b);
        if (pa == pb) {
            return false;
        }
        if (rank[pa] < rank[pb]) {
            parent[pa] = pb;
        } else if (rank[pa] > rank[pb]) {
            parent[pb] = pa;
        } else {
            parent[pb] = pa;
            rank[pa]++;
        }
        return true;
    }
}

class MSTResult {
    public final int totalWeight;
    public final List<KruskalEdge> edges;

    MSTResult(int totalWeight, List<KruskalEdge> edges) {
        this.totalWeight = totalWeight;
        this.edges = Collections.unmodifiableList(new ArrayList<>(edges));
    }
}
