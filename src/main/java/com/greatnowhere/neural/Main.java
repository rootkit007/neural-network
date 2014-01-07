package com.greatnowhere.neural;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;

import javax.imageio.ImageIO;

/**
 * Main runnable class
 * @author pzeltins
 *
 */
public class Main {

	static NetworkTrainer t = new NetworkTrainer();
	
	final static int WIDTH = 50;
	final static int HEIGHT = 50;
	final static int HIDDEN = 600;
	final static String networkPersistencePath = "src/test/resources/network.file";
	
	public static void main(String[] args) throws InterruptedException, IOException {
		try {
			t.init(networkPersistencePath, 2D, 0.4D); // init with default values
		} catch (Exception ex) {
			System.out.println("cannot read network from file " + ex.getLocalizedMessage());
			t.init(3 * WIDTH * HEIGHT, 1, HIDDEN, 2D, 0.4D);
		}

		processImages("src/test/resources/vehicles", 1D);
		processImages("src/test/resources/background", 0D);
		do {
			t.train(networkPersistencePath);
			System.out.println();
			System.out.println("Network max error = " + t.n.maxError);
		} while ( t.n.maxError > 0.05D);
		
	}

	/**
	 * Adds specified image and its associated target value to training set
	 * Image is first resized, then flat array of image's RGB values is constructed
	 * @param img
	 * @param targetValue
	 * @param label
	 */
	static void processImage(BufferedImage img, double targetValue,String label) {
		BufferedImage resizedImg = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = resizedImg.createGraphics();
		g.drawImage(img, 0, 0, WIDTH, HEIGHT, null);
		g.dispose();
		double[] pixels = new double[3 * WIDTH * HEIGHT];
		int idx = 0;
		for (int x=0;x<WIDTH;x++) {
			for (int y=0;y<HEIGHT;y++) {
				pixels[idx++] = ( img.getRGB(x, y) & 0xff ) / 255D;
				pixels[idx++] = (( img.getRGB(x, y) >> 8 ) & 0xff) / 255D;
				pixels[idx++] = (img.getRGB(x, y) >> 16) / 255D;
			}
		}
		t.addTrainingSet(pixels, targetValue, label);
	}
	
	/**
	 * Adds images from specified path to training set
	 * @param path
	 * @param targetValue
	 */
	static void processImages(String path, double targetValue) {
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

	
}
