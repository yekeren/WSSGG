import java.io.*;
import java.util.*;

import edu.stanford.nlp.scenegraph.RuleBasedParser;
import edu.stanford.nlp.scenegraph.SceneGraph;

public class SceneGraphDemo {
  public static void main(String[] args) throws IOException {
    Scanner scanner = new Scanner(System.in);
    RuleBasedParser parser = new RuleBasedParser();

    String line;
    while ((line = scanner.nextLine()) != null) {
      SceneGraph scene = parser.parse(line);

      //printing the scene graph in a readable format
      System.out.println(scene.toReadableString()); 
      
      //printing the scene graph in JSON form
      System.out.println(scene.toJSON(0, "", "")); 
    }
    System.out.println("Done");
  }
}
