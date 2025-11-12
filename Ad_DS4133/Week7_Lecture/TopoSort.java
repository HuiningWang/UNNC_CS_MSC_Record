import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

public class TopoSort {
    public static List<Integer> topoSort(int n, List<int[]> edges) {
        List<List<Integer>> g = new ArrayList<>();
        int[] indeg = new int[n];

        for (int i = 0; i < n; i++) {
            g.add(new ArrayList<>());
        }

        for (int[] e : edges) {
            int u = e[0], v = e[1];
            g.get(u).add(v);
            indeg[v]++;
        }

        Deque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (indeg[i] == 0) {
                q.add(i);
            }
        }

        List<Integer> order = new ArrayList<>();

        while (!q.isEmpty()) {
            int u = q.poll();
            order.add(u);
            for (int v : g.get(u)) {
                if (--indeg[v] == 0) {
                    q.add(v);
                }
            }
        }
        if (order.size() != n) {
            throw new IllegalArgumentException("not a DAG");
        }

        return order;

    }

    public static void main(String[] args) {
        int n = 6;
        List<int[]> edges = new ArrayList<>();
        edges.add(new int[] { 5, 2 });
        edges.add(new int[] { 5, 0 });
        edges.add(new int[] { 4, 0 });
        edges.add(new int[] { 4, 1 });
        edges.add(new int[] { 2, 3 });
        edges.add(new int[] { 3, 1 });

        List<Integer> order = topoSort(n, edges);
        System.out.println(order);
    }
}
