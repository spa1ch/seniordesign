import java.util.Random;
import edu.mines.jtk.util.*;

/**
 * Rates for each method
 *
 * Transpose13S1 (Serial):
 * (n1,n2,n3) = (101)(102)(103), rate = 170
 * (n1,n2,n3) = (201)(202)(203), rate = 77
 * (n1,n2,n3) = (301)(302)(303), rate = 37
 * (n1,n2,n3) = (401)(402)(403), rate = 41
 * Transpose13P1 (Parallel):
 * (n1,n2,n3) = (101)(102)(103), rate = 418
 * (n1,n2,n3) = (201)(202)(203), rate = 115
 * (n1,n2,n3) = (301)(302)(303), rate = 90
 * (n1,n2,n3) = (401)(402)(403), rate = 92
 * Transpose13S8 (Serial):
 * (n1,n2,n3) = (101)(102)(103), rate = 367
 * (n1,n2,n3) = (201)(202)(203), rate = 191
 * (n1,n2,n3) = (301)(302)(303), rate = 174
 * (n1,n2,n3) = (401)(402)(403), rate = 152
 * Transpose13P8 (Parallel):
 * (n1,n2,n3) = (101)(102)(103), rate = 733
 * (n1,n2,n3) = (201)(202)(203), rate = 388
 * (n1,n2,n3) = (301)(302)(303), rate = 397
 * (n1,n2,n3) = (401)(402)(403), rate = 464
 * Copy (Parallel):
 * (n1,n2,n3) = (101)(102)(103), rate = 1479
 * (n1,n2,n3) = (201)(202)(203), rate = 1136
 * (n1,n2,n3) = (301)(302)(303), rate = 1293
 * (n1,n2,n3) = (401)(402)(403), rate = 1228
 */

public class TransposeBench {

  public static void main(String[] args) {
    Method ts = new Method() {
      public void apply(float[][][] x, float[][][] y){
        transpose13S1(x,y);
    }};
    Method tp = new Method() {
      public void apply(float[][][] x, float[][][] y){
        transpose13P1(x,y);
    }};
    Method ts8 = new Method() {
      public void apply(float[][][] x, float[][][] y){
        transpose13S8(x,y);
    }};
    Method tp8 = new Method() {
      public void apply(float[][][] x, float[][][] y){
        transpose13P8(x,y);
    }};
    Method cp = new Method() {
      public void apply(float[][][] x, float[][][] y){
        copyP(x,y);
    }};
    Method[] methods = {ts,tp,ts8,tp8};
    String[] names = {"Transpose13S1 (Serial):","Transpose13P1 (Parallel):",
                      "Transpose13S8 (Serial):","Transpose13P8 (Parallel):"};
    for (int i = 0; i<methods.length; ++i)
      benchTranspose(methods[i],names[i]);
    Method[] methods2 = {cp};
    String[] names2 = {"Copy (Parallel):"};
    for (int i = 0; i<methods2.length; ++i)
      benchCopy(methods2[i],names2[i]);
  }


/*********************TRANSPOSE*********************/

  private static void benchTranspose(Method method, String name) {
    System.out.println(name);
    for (int n = 100; n<=400; n+=100) {
      float[][][] x = randFloArray(n+1,n+2,n+3);
      float[][][] y = new float[n+3][n+2][n+1];
      double maxtime = 2.0;
      int ntrans;
      Stopwatch sw = new Stopwatch();
      sw.restart();
      for (ntrans = 0; sw.time()<maxtime; ++ntrans)
        method.apply(x,y);
      sw.stop();
      for (int i3 = 0; i3<n+1; ++i3) 
        for (int i2 = 0; i2<n+1; ++i2) 
          for (int i1 = 0; i1<n+2; ++i1) 
	    assert y[i1][i2][i3]==x[i3][i2][i1]:"y == x";
      int rate = (int)(1.0e-6*ntrans*(n+1)*(n+2)*(n+3)/sw.time());
      System.out.println("(n1,n2,n3) = ("+(n+1)+")("+(n+2)+")("+(n+3)+"), rate = "+rate);
    }
  }

  private static void transposeS(float[][] x, float[][] y) {
    int n1 = x[0].length;
    int n2 = x.length;
    for (int i1 = 0; i1<n1; ++i1) 
      for (int i2 =0 ; i2<n2; ++i2)
        y[i1][i2] = x[i2][i1];
  }

  private static void transposeP(final float[][] x, final float[][] y) {
    final int n1 = x[0].length;
    final int n2 = x.length;
    Parallel.loop(n1,new Parallel.LoopInt() {
      public void compute(int i1) {
        for (int i2 =0 ; i2<n2; ++i2)
          y[i1][i2] = x[i2][i1];
      }
    });
  }

  private static void transpose13S1(float[][][] x, float[][][] y) {
    int n1 = x[0][0].length;
    int n2 = x[0].length;
    int n3 = x.length;
    for (int i1 = 0; i1<n1; ++i1) 
      for (int i2 =0 ; i2<n2; ++i2)
        for (int i3 = 0; i3<n3; ++i3) 
          y[i1][i2][i3] = x[i3][i2][i1];
  }

  private static void transpose13S8(float[][][] x, float[][][] y) {
    int n1 = x[0][0].length;
    int n2 = x[0].length;
    int n3 = x.length;
    int rem = n1%8;
    int k1;
    for (k1 = 0; k1<rem; ++k1) 
      for (int k2 =0 ; k2<n2; ++k2)
        for (int k3 = 0; k3<n3; ++k3) 
          y[k1][k2][k3] = x[k3][k2][k1];
    for (int i1 = k1; i1<n1; i1+=8) 
      for (int i2 =0 ; i2<n2; ++i2)
        for (int i3 = 0; i3<n3; ++i3) {
          y[i1  ][i2][i3] = x[i3][i2][i1  ];
          y[i1+1][i2][i3] = x[i3][i2][i1+1];
          y[i1+2][i2][i3] = x[i3][i2][i1+2];
          y[i1+3][i2][i3] = x[i3][i2][i1+3];
          y[i1+4][i2][i3] = x[i3][i2][i1+4];
          y[i1+5][i2][i3] = x[i3][i2][i1+5];
          y[i1+6][i2][i3] = x[i3][i2][i1+6];
          y[i1+7][i2][i3] = x[i3][i2][i1+7];
	}
  }

  private static void transpose13P1(final float[][][] x, final float[][][] y) {
    final int n1 = x[0][0].length;
    final int n2 = x[0].length;
    final int n3 = x.length;
    Parallel.loop(n1,new Parallel.LoopInt() {
      public void compute(int i1) {
        for (int i2 =0 ; i2<n2; ++i2)
          for (int i3 =0 ; i3<n3; ++i3)
            y[i1][i2][i3] = x[i3][i2][i1];
      }
    });
  }

  private static void transpose13P8(final float[][][] x, final float[][][] y) {
    final int n1 = x[0][0].length;
    final int n2 = x[0].length;
    final int n3 = x.length;
    int rem = n1%16;
    int k1;
    for (k1 = 0; k1<rem; ++k1) 
      for (int k2 =0 ; k2<n2; ++k2)
        for (int k3 = 0; k3<n3; ++k3) 
          y[k1][k2][k3] = x[k3][k2][k1];
    Parallel.loop(k1,n1,16,new Parallel.LoopInt() {
      public void compute(int i1) {
        for (int i2 =0 ; i2<n2; ++i2)
          for (int i3 = 0; i3<n3; ++i3) {
            y[i1  ][i2][i3] = x[i3][i2][i1  ];
            y[i1+1][i2][i3] = x[i3][i2][i1+1];
            y[i1+2][i2][i3] = x[i3][i2][i1+2];
            y[i1+3][i2][i3] = x[i3][i2][i1+3];
            y[i1+4][i2][i3] = x[i3][i2][i1+4];
            y[i1+5][i2][i3] = x[i3][i2][i1+5];
            y[i1+6][i2][i3] = x[i3][i2][i1+6];
            y[i1+7][i2][i3] = x[i3][i2][i1+7];
            y[i1+8][i2][i3] = x[i3][i2][i1+8];
            y[i1+9][i2][i3] = x[i3][i2][i1+9];
            y[i1+10][i2][i3] = x[i3][i2][i1+10];
            y[i1+11][i2][i3] = x[i3][i2][i1+11];
            y[i1+12][i2][i3] = x[i3][i2][i1+12];
            y[i1+13][i2][i3] = x[i3][i2][i1+13];
            y[i1+14][i2][i3] = x[i3][i2][i1+14];
            y[i1+15][i2][i3] = x[i3][i2][i1+15];
	}
      }
    });
  }


/*********************COPY*********************/

  private static void benchCopy(Method method, String name) {
    System.out.println(name);
    for (int n = 100; n<=400; n+=100) {
      float[][][] x = randFloArray(n+1,n+2,n+3);
      float[][][] y = new float[n+1][n+2][n+3];
      double maxtime = 2.0;
      int ncopy;
      Stopwatch sw = new Stopwatch();
      sw.restart();
      for (ncopy = 0; sw.time()<maxtime; ++ncopy)
        method.apply(x,y);
      sw.stop();
      for (int i3 = 0; i3<n+1; ++i3) 
        for (int i2 = 0; i2<n+1; ++i2) 
          for (int i1 = 0; i1<n+2; ++i1) 
	    assert y[i3][i2][i1]==x[i3][i2][i1]:"y == x";
      int rate = (int)(1.0e-6*ncopy*(n+1)*(n+2)*(n+3)/sw.time());
      System.out.println("(n1,n2,n3) = ("+(n+1)+")("+(n+2)+")("+(n+3)+"), rate = "+rate);
    }
  }
 
  private static void copyS(float[][] x, float[][] y) {
    int n1 = x[0].length;
    int n2 = x.length;
    for (int i2 = 0; i2<n2; ++i2) 
      for (int i1 = 0; i1<n1; ++i1) 
        y[i2][i1] = x[i2][i1];
  }

  // slower than serial version
  private static void copyP(final float[][] x, final float[][] y) {
    final int n1 = x[0].length;
    final int n2 = x.length;
    Parallel.loop(n2,new Parallel.LoopInt() {
      public void compute(int i2) {
        for (int i1 = 0; i1<n1; ++i1) 
          y[i2][i1] = x[i2][i1];
      }
    });
  }

  private static void copyS(float[][][] x, float[][][] y) {
    int n1 = x[0][0].length;
    int n2 = x[0].length;
    int n3 = x.length;
    for (int i3 = 0; i3<n3; ++i3)
      copyS(x[i3],y[i3]);
  }

  // quicker than serial version
  private static void copyP(final float[][][] x, final float[][][] y) {
    final int n1 = x[0][0].length;
    final int n2 = x[0].length;
    final int n3 = x.length;
    Parallel.loop(n3,new Parallel.LoopInt() {
      public void compute(int i3) {
        copyS(x[i3],y[i3]);
      }
    });
  }
 

/*********************UTILITIES*********************/

  private interface Method {
    public void apply(float[][][] x, float[][][] y);
  }

  private static float[][] randFloArray(int n2, int n1) {
    Random r = new Random();
    float[][] x = new float[n2][n1];
    for (int i2 = 0; i2<n2; ++i2)
      for (int i1 = 0; i1<n1; ++i1)
        x[i2][i1] = r.nextFloat();
    return x;
  }
  
  private static float[][][] randFloArray(int n3, int n2, int n1) {
    Random r = new Random();
    float [][][] x = new float[n3][n2][n1];
    for (int i3 = 0; i3<n3; ++i3)
      for (int i2 = 0; i2<n2; ++i2)
        for (int i1 = 0; i1<n1; ++i1)
          x[i3][i2][i1] = r.nextFloat();
    return x;
  }


}
