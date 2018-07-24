package front;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class Attendance implements ActionListener {
	JFrame frame3;
	JPanel p1;
	JPanel p2;
	JPanel p3;
	JLabel l1;
	JLabel l2;
	JLabel l3;
	JLabel l4;
	JLabel l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16;
	JButton b1;
	JButton b2;
	Attendance()
	{
		frame3=new JFrame("Attendance");
		p1= new JPanel(new FlowLayout());
		p2= new JPanel(new GridLayout(7,2));
		p3= new JPanel(new FlowLayout());
		p1.setBorder(BorderFactory.createLineBorder(Color.black));
		p2.setBorder(BorderFactory.createLineBorder(Color.black));
		p3.setBorder(BorderFactory.createLineBorder(Color.black));
		l1 = new JLabel("<html><span style='font-size:25px; color:Red;'>"+"Attendance"+"</span></html>");
		l2 = new JLabel("OS");
		l3 = new JLabel("XCS");
		l4 = new JLabel("JAVA");
		l5 = new JLabel("DAA");
		l6 = new JLabel("Compiler");
		l7 = new JLabel("Tech PDP");
		l8 = new JLabel("XCS Lab");
		l9 = new JLabel("          ");
		l10 = new JLabel("          ");
		l11 = new JLabel("          ");
		l12 = new JLabel("          ");
		l13 = new JLabel("          ");
		l14 = new JLabel("          ");
		l15 = new JLabel("          ");
		//l16 = new JLabel("          ");
		//l17= new JLabel("          ");
		//l18= new JLabel("          ");
		l1.setHorizontalAlignment(JLabel.CENTER);
		l2.setHorizontalAlignment(JLabel.CENTER);
		l3.setHorizontalAlignment(JLabel.CENTER);
		l4.setHorizontalAlignment(JLabel.CENTER);
		l5.setHorizontalAlignment(JLabel.CENTER);
		l6.setHorizontalAlignment(JLabel.CENTER);
		l7.setHorizontalAlignment(JLabel.CENTER);
		l8.setHorizontalAlignment(JLabel.CENTER);
		b1= new JButton("Back");
		b2= new JButton("Home");
		b1.addActionListener (this);
		b2.addActionListener (this);
		p1.add(l1);
		p2.add(l2);
		p2.add(l9);
		p2.add(l3);
		p2.add(l10);
		p2.add(l4);
		p2.add(l11);
		p2.add(l5);
		p2.add(l12);
		p2.add(l6);
		p2.add(l13);
		p2.add(l7);
		p2.add(l14);
		p2.add(l8);
		p2.add(l15);
		p3.add(b1);
		p3.add(b2);
		frame3.add(p1);
		frame3.add(p2);
		frame3.add(p3);
		frame3.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame3.setLayout(new GridLayout(3,1));
		frame3.setSize(500,400);
		frame3.setLocation(200, 100);
		frame3.setVisible(true);
		
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		
		if(e.getSource()==b1)
		{
			frame3.setVisible(false);
			new Option();
		}
		else if(e.getSource()==b2)
		{
			frame3.setVisible(false);
			new Start();
		}
	}
}
