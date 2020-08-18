import java.io.*;
import java.util.*;

import edu.stanford.nlp.scenegraph.RuleBasedParser;
import edu.stanford.nlp.scenegraph.SceneGraph;

public class SceneGraphDemo {

  public static void main(String[] args) throws IOException {

    String line;
    Scanner scanner = new Scanner(System.in);
    RuleBasedParser parser = new RuleBasedParser();

    System.err.println("Processing from stdin. Enter one sentence per line.");
    System.err.print("> ");

    Integer count = 0;
    while (scanner.hasNextLine()) {
      line = scanner.nextLine();
      SceneGraph scene = parser.parse(line);

      if (scene != null) {
        // Printing the scene graph in a readable format.
        System.err.println(scene.toReadableString()); 

        // Printing the scene graph in JSON form.
        // Parameteres are id, url, and phrase.
        System.out.println(scene.toJSON(count, "", line)); 
      } else {
        System.err.printf("{\"phrase\": \"%s\", \"id\": %d}\n", line, count);
        System.out.printf("{\"phrase\": \"%s\", \"id\": %d}\n", line, count);
      }

      System.err.print("> ");
      System.err.flush();
      System.out.flush();
      count += 1;
    }
  }

}
