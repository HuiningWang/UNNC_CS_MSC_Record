import java.util.*;

public class PrimMST {
    public static MSTResult prim(int n, List<int[]> edges) {
        if (n <= 0) {
            throw new IllegalArgumentException("Number of vertices must be positive");
        }

        List<List<Edge>> g = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            g.add(new ArrayList<>());
        }
        for (int[] e : edges) {
            if (e.length != 3) {
                throw new IllegalArgumentException("Edge must contain exactly 3 elements: u, v, w");
            }
            int u = e[0], v = e[1], w = e[2];
            if (u < 0 || u >= n || v < 0 || v >= n) {
                throw new IllegalArgumentException("Vertex index out of range");
            }
            g.get(u).add(new Edge(v, w));
            g.get(v).add(new Edge(u, w));
        }

        boolean[] vis = new boolean[n];
        int[] bestWeight = new int[n];
        Arrays.fill(bestWeight, Integer.MAX_VALUE);

        PriorityQueue<State> pq = new PriorityQueue<>(Comparator.comparingInt(s -> s.weight));
        pq.offer(new State(0, -1, 0));
        bestWeight[0] = 0;

        int totalWeight = 0;
        int count = 0;
        List<int[]> treeEdges = new ArrayList<>();

        while (!pq.isEmpty() && count < n) {
            State cur = pq.poll();
            int u = cur.vertex;
            if (vis[u]) {
                continue;
            }
            vis[u] = true;
            totalWeight += cur.weight;
            if (cur.parent != -1) {
                treeEdges.add(new int[] { cur.parent, u, cur.weight });
            }
            count++;

            for (Edge edge : g.get(u)) {
                if (!vis[edge.to] && edge.w < bestWeight[edge.to]) {
                    bestWeight[edge.to] = edge.w;
                    pq.offer(new State(edge.to, u, edge.w));
                }
            }
        }
        if (count != n) {
            throw new IllegalStateException("Graph not connected");
        }
        return new MSTResult(totalWeight, treeEdges);
    }

    public static void main(String[] args) {
        int n = 4;
        List<int[]> edges = new ArrayList<>();
        edges.add(new int[] { 0, 1, 2 });
        edges.add(new int[] { 0, 2, 1 });
        edges.add(new int[] { 0, 3, 3 });
        edges.add(new int[] { 2, 3, 4 });
        edges.add(new int[] { 1, 3, 5 });

        MSTResult ans = prim(n, edges);
        System.out.println("MST total weight = " + ans.totalWeight);
        System.out.println("MST edges:");
        for (int[] e : ans.edges) {
            System.out.println(e[0] + " - " + e[1] + " (w=" + e[2] + ")");
        }
    }

    private static class Edge {
        final int to;
        final int w;

        Edge(int to, int w) {
            this.to = to;
            this.w = w;
        }
    }

    public static class MSTResult {
        public final int totalWeight;
        public final List<int[]> edges;

        MSTResult(int totalWeight, List<int[]> edges) {
            this.totalWeight = totalWeight;
            this.edges = Collections.unmodifiableList(edges);
        }
    }

    private static class State {
        final int vertex;
        final int parent;
        final int weight;

        State(int vertex, int parent, int weight) {
            this.vertex = vertex;
            this.parent = parent;
            this.weight = weight;
        }
    }
}