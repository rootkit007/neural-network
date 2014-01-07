/*
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;

import javax.imageio.ImageIO;

import org.junit.Ignore;
import org.junit.Test;
*/
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.greatnowhere.neural.NetworkTrainer;

@RunWith(JUnit4.class)
public class CarRecognitionTest {

	NetworkTrainer t = new NetworkTrainer();
	
	final static int WIDTH = 20;
	final static int HEIGHT = 20;
	final static int HIDDEN = 20;
	final static String networkPersistencePath = "src/test/resources/network.file";

	/*
	@Test
	@Ignore
	public void test() throws InterruptedException, IOException {
		try {
			t.init(networkPersistencePath, 0.6D, 0.4D);
		} catch (Exception ex) {
			System.out.println("cannot read network from file " + ex.getLocalizedMessage());
			t.init(3 * WIDTH * HEIGHT, 1, HIDDEN, 2D, 0.4D);
		}

		processImages("src/test/resources/vehicles", 1D);
		processImages("src/test/resources/background", 0D);
		double successRate = 0D;
		do {
			for (int i=0; i<10; i++) {
				successRate = t.train(networkPersistencePath);
				System.out.println("success rate " + successRate);
			}
			if ( successRate < 0.9D ) {
				t.n.addHiddenNeuron(1);
				System.out.println("adding hidden layer neuron");
			}
		} while ( t.maxError > 0.05D);
		
	}

	void processImage(BufferedImage img, double targetValue,String label) {
		BufferedImage resizedImg = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = resizedImg.createGraphics();
		g.drawImage(img, 0, 0, WIDTH, HEIGHT, null);
		g.dispose();
		double[] pixels = new double[3 * WIDTH * HEIGHT];
		int idx = 0;
		for (int x=0;x<WIDTH;x++) {
			for (int y=0;y<HEIGHT;y++) {
				int rgb = img.getRGB(x, y);
				pixels[idx++] = rgb & 0xFF;
				pixels[idx++] = ( rgb >> 8 ) & 0xFF;
				pixels[idx++] = ( rgb >> 16 ) &0xFF;
			}
		}
		t.addTrainingSet(pixels, targetValue, label);
	}
	
	void processImages(String path, double targetValue) {
		Path imgpath = FileSystems.getDefault().getPath(path);
		try (
				DirectoryStream<Path> imgdir = Files.newDirectoryStream(imgpath);
		) {
			for(Path file : imgdir ) {
				File f = file.toFile();
				if ( f.isFile() && f.canRead() ) {
					BufferedImage img = ImageIO.read(f);
					processImage(img, targetValue, f.getName());
					System.out.println("added image " + f.getName());
				}
			}
		} catch (Exception ex) {
			System.out.println(ex.getLocalizedMessage());
		}
		
	}
	*/
	
}
