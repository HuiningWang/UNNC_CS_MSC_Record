import java.util.*;

public class FloydWarshallSolver {
    public static class FloydResult {
        public final double[][] dist;
        public final int[][] next;

        FloydResult(double[][] dist, int[][] next) {
            this.dist = dist;
            this.next = next;
        }
    }

    public static FloydResult floydWarshall(int n, List<int[]> edges, boolean directed) {
        if (n <= 0) {
            throw new IllegalArgumentException("Number of vertices must be positive");
        }
        Objects.requireNonNull(edges, "Edges list must not be null");

        final double INF = 1e9;
        double[][] dist = new double[n][n];
        int[][] next = new int[n][n];

        for (int i = 0; i < n; i++) {
            Arrays.fill(dist[i], INF);
            Arrays.fill(next[i], -1);
            dist[i][i] = 0;
        }

        for (int[] e : edges) {
            if (e == null || e.length != 3) {
                throw new IllegalArgumentException("Edge must contain u, v, w");
            }
            int u = e[0];
            int v = e[1];
            double w = e[2];
            if (u < 0 || u >= n || v < 0 || v >= n) {
                throw new IllegalArgumentException("Vertex index out of range");
            }
            if (w < dist[u][v]) {
                dist[u][v] = w;
                next[u][v] = v;
            }
            if (!directed && w < dist[v][u]) {
                dist[v][u] = w;
                next[v][u] = u;
            }
        }

        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                        next[i][j] = next[i][k];
                    }
                }
            }
        }

        for (int i = 0; i < n; i++) {
            if (dist[i][i] < 0) {
                throw new IllegalStateException("Graph contains a negative weight cycle");
            }
        }

        return new FloydResult(dist, next);
    }

    public static List<Integer> getPath(int[][] next, int u, int v) {
        if (next[u][v] == -1) {
            return Collections.emptyList();
        }
        List<Integer> path = new ArrayList<>();
        path.add(u);
        while (u != v) {
            u = next[u][v];
            if (u == -1) {
                return Collections.emptyList();
            }
            path.add(u);
        }
        return path;
    }

    public static void main(String[] args) {
        int n = 4;
        List<int[]> edges = new ArrayList<>();
        edges.add(new int[] { 0, 1, 5 });
        edges.add(new int[] { 0, 3, 9 });
        edges.add(new int[] { 1, 2, 2 });
        edges.add(new int[] { 2, 0, 3 });
        edges.add(new int[] { 2, 3, 4 });

        FloydResult res = floydWarshall(n, edges, true);

        System.out.println("Shortest distance matrix:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (res.dist[i][j] >= 1e9) {
                    System.out.print("INF ");
                } else {
                    System.out.printf("%3.0f ", res.dist[i][j]);
                }
            }
            System.out.println();
        }

        System.out.println();
        System.out.println("Example path 1 -> 3:");
        System.out.println(getPath(res.next, 1, 3));
    }
}
